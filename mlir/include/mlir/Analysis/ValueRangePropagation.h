//===- ValueRangePropagation.cpp - Value Range Analysis -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Value range analysis attempts to compute the range of concrete values
// that every abstract value (i.e., mlir::Value) can take at runtime.
// The analysis does not make assumptions about the type of values.
// The analysis itself is a wrapper around DataFlowAnalysis, with the key
// trick for convergence being a widening operator, defined in
// VRPAnalysisBase::visitOperation.
//
// How to use:
//   VRPAnalysis<VRPImplBase> provides useful results over some standard
//   operations and types.
//   Clients that want precision over more operations and types must extend
//   VRPImplBase and define comparison operations for the types and transfer
//   functions (rangeFold) for the operations. VRPAnalysis can then be
//   instantiated with this extension.
//
//===----------------------------------------------------------------------===//

namespace mlir {

// Extention of Attribute with possibly +INF/-INF values.
// Type parameter AttrCmp must implement
//     static llvm::Optional<bool>
//       AttrCmp::cmpLT(const Attribute &lhs, const Attribute &rhs);
template <typename AttrCmp>
class VRPAttribute : public Attribute {

  enum InfinityStatus {
    NotINF,   // Not infinity
    PlusINF,  // +INF
    MinusINF, // -INF
  } infinityStatus;

  VRPAttribute(InfinityStatus is) : infinityStatus(is){};

public:
  VRPAttribute(const Attribute &v) : Attribute(v), infinityStatus(NotINF){};
  static VRPAttribute getPInfinity() { return VRPAttribute(PlusINF); }
  static VRPAttribute getMInfinity() { return VRPAttribute(MinusINF); }
  Attribute getValue() const {
    assert(!isInfinity());
    return static_cast<Attribute>(*this);
  }

  bool isInfinity() const { return infinityStatus != NotINF; }
  bool isPInfinity() const { return infinityStatus == PlusINF; }
  bool isMInfinity() const { return infinityStatus == MinusINF; }

  bool operator==(const VRPAttribute &rhs) const {
    return infinityStatus == rhs.infinityStatus && Attribute::operator==(rhs);
  }
  bool operator!=(const VRPAttribute &rhs) const { return !operator==(rhs); }

  llvm::Optional<bool> cmpLT(const VRPAttribute &rhs) const {
    if ((isMInfinity() && rhs.isMInfinity()) ||
        (isPInfinity() && rhs.isPInfinity())) {
      return {false};
    }
    if ((isMInfinity() || rhs.isPInfinity())) {
      return {true};
    }
    if (isPInfinity() || rhs.isMInfinity())
      return {false};

    assert(!isInfinity() && !rhs.isInfinity());

    return AttrCmp::cmpLT(*this, rhs);
  }

  llvm::Optional<bool> cmpLE(const VRPAttribute &rhs) const {
    return cmpLT(rhs).map([this, &rhs](bool lt) { return lt || *this == rhs; });
  };

  llvm::Optional<bool> cmpGT(const VRPAttribute &rhs) const {
    return cmpLE(rhs).map([](bool le) { return !le; });
  };
  llvm::Optional<bool> cmpGE(const VRPAttribute &rhs) const {
    return cmpGT(rhs).map([this, &rhs](bool gt) { return gt || *this == rhs; });
  };

  static llvm::Optional<VRPAttribute> min(const VRPAttribute &lhs,
                                          const VRPAttribute &rhs) {
    return lhs.cmpLE(rhs).map([&lhs, &rhs](bool le) { return le ? lhs : rhs; });
  }

  static llvm::Optional<VRPAttribute> max(const VRPAttribute &lhs,
                                          const VRPAttribute &rhs) {
    return lhs.cmpLE(rhs).map([&lhs, &rhs](bool le) { return le ? rhs : lhs; });
  }

  void print(raw_ostream &os) const {
    if (isPInfinity()) {
      os << "INF";
    } else if (isMInfinity()) {
      os << "-INF";
    } else {
      Attribute::print(os);
    }
  };
  void dump() const { print(llvm::errs()); };
};

template <typename AttrCmp>
using VRPRange = std::pair<VRPAttribute<AttrCmp>, VRPAttribute<AttrCmp>>;

// mlir::LatticeElement for the data-flow analysis.
template <typename AttrCmp>
class VRPLatticeEl {
  using VT = VRPAttribute<AttrCmp>;
  VRPRange<AttrCmp> range;

public:
  bool validate() const {
    return range.first.cmpLE(range.second).getValueOr(false);
  };
  VRPLatticeEl(VRPRange<AttrCmp> r) : range(r) { assert(validate()); }
  VRPLatticeEl()
      : range(std::make_pair(VT::getMInfinity(), VT::getPInfinity())) {
    assert(validate());
  }
  VRPRange<AttrCmp> getRange() const { return range; }

  /// Satisfying the interface for LatticeElement

  static VRPLatticeEl getPessimisticValueState(MLIRContext *context) {
    return VRPLatticeEl();
  }
  static VRPLatticeEl getPessimisticValueState(Value value) {
    return VRPLatticeEl();
  }
  bool operator==(const VRPLatticeEl &rhs) const {
    return range.first == rhs.range.first && range.second == rhs.range.second;
  }
  static VRPLatticeEl join(const VRPLatticeEl &lhs, const VRPLatticeEl &rhs) {
    return VRPLatticeEl(
        std::make_pair(VT::min(lhs.range.first, rhs.range.first)
                           .getValueOr(VT::getMInfinity()),
                       VT::max(lhs.range.second, rhs.range.second)
                           .getValueOr(VT::getPInfinity())));
  }

  void print(raw_ostream &os) const {
    os << "[";
    range.first.print(os);
    os << " ; ";
    range.second.print(os);
    os << "]";
  }
  void dump() const { print(llvm::errs()); };
};

// The analyis must be instantiated with a class that defines:
// 1. A comparison operator over Attributes.
//     static llvm::Optional<bool>
//       AttrCmp::cmpLT(const Attribute &lhs, const Attribute &rhs);
// 2. Similar to operation::fold, this implements a fold over VRPRange,
//    describing the range of output values, given the ranges for each
//    input operand. failure() can be returned if nothing useful could
//    be computed.
//      static LogicalResult
//        rangeFold(Operation *op, ArrayRef<VRPRange<AttrCmpRangeFold>>
//          operands, SmallVectorImpl<VRPRange<AttrCmpRangeFold>> &results);
template <typename AttrCmpRangeFold>
struct VRPAnalysis
    : public ForwardDataFlowAnalysis<VRPLatticeEl<AttrCmpRangeFold>> {
  VRPAnalysis(mlir::MLIRContext *c)
      : ForwardDataFlowAnalysis<VRPLatticeEl<AttrCmpRangeFold>>(c) {}
  ~VRPAnalysis() override = default;

  void print(Operation *topLevelOp, raw_ostream &os) {
    topLevelOp->walk([&os, this](mlir::Operation *op) {
      for (Value result : op->getOpResults()) {
        LatticeElement<VRPLatticeEl<AttrCmpRangeFold>> *lattice =
            this->lookupLatticeElement(result);
        if (lattice && !lattice->isUninitialized()) {
          os << result << " : ";
          lattice->getValue().print(os);
          os << "\n";
        }
      }
    });
  }
  void dump() const { print(llvm::errs()); };

protected:
  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<VRPLatticeEl<AttrCmpRangeFold>> *>
                     operands) final {
    using VRPV = VRPAttribute<AttrCmpRangeFold>;
    using VRPR = VRPRange<AttrCmpRangeFold>;
    SmallVector<VRPR> opdsRange(llvm::map_range(
        operands, [](LatticeElement<VRPLatticeEl<AttrCmpRangeFold>> *value) {
          return value->getValue().getRange();
        }));

    SmallVector<VRPR> foldResults;
    if (failed(AttrCmpRangeFold::rangeFold(op, opdsRange, foldResults))) {
      return this->markAllPessimisticFixpoint(op->getResults());
    }

    // Merge the fold results into the lattice for this operation.
    assert(foldResults.size() == op->getNumResults() && "invalid result size");
    ChangeResult result = ChangeResult::NoChange;
    for (unsigned i = 0, e = foldResults.size(); i != e; ++i) {
      LatticeElement<VRPLatticeEl<AttrCmpRangeFold>> &lattice =
          this->getLatticeElement(op->getResult(i));

      // The VRP lattice doesn't satisfy the ascending chain condition.
      // So we need a widening operator.
      // See "Static Determination of Dynamic Properties of Programs"
      // - Cousot & Cousot
      if (!lattice.isUninitialized()) {
        VRPR prevResult = lattice.getValue().getRange();
        if (foldResults[i].first.cmpLT(prevResult.first).getValueOr(true)) {
          foldResults[i].first = VRPV::getMInfinity();
        }
        if (foldResults[i].second.cmpGT(prevResult.second).getValueOr(true)) {
          foldResults[i].second = VRPV::getPInfinity();
        }
      }
      result |= lattice.join(foldResults[i]);
    }
    return result;
  }
};

// This class implements a comparison operator over Attributes and a range
// fold function (i.e., the transfer function for the data-flow analysis)
// for known types and operations. Clients can extend as necessary.
struct VRPImplBase {
  static llvm::Optional<bool> cmpLT(const Attribute &lhs, const Attribute &rhs);
  static LogicalResult
  rangeFold(Operation *op, ArrayRef<VRPRange<VRPImplBase>> operands,
            SmallVectorImpl<VRPRange<VRPImplBase>> &results);
};

} // namespace mlir
