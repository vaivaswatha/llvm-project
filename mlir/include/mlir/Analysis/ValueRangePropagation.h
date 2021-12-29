//===- VRP.cpp - Value Range Analysis -----------------===//
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
//
//===----------------------------------------------------------------------===//

namespace mlir {

// We compute a range <VRPValue, VRPValue> for each mlir::Value.
// The type parameter T must have the following methods defined:
//   llvm::Optional<bool> cmpLT(const T &rhs) const
//   static bool operator==(const T &lhs, const T &rhs)
//   void print(raw_ostream &os) const;
// The compare less-than function (cmpLT) returns None when
// the the values are not comparable.
template <typename T>
class VRPValue {

  enum InfinityStatus {
    NotINF,   // Not infinity
    PlusINF,  // +INF
    MinusINF, // -INF
  } infinityStatus;
  T value;

  VRPValue(InfinityStatus is) : infinityStatus(is), value({}){};

public:
  VRPValue(const T &v) : infinityStatus(NotINF), value(v){};
  static VRPValue getPInfinity() { return VRPValue(PlusINF); }
  static VRPValue getMInfinity() { return VRPValue(MinusINF); }
  T getValue() const {
    assert(!isInfinity());
    return value;
  }

  bool isInfinity() const { return infinityStatus != NotINF; }
  bool isPInfinity() const { return infinityStatus == PlusINF; }
  bool isMInfinity() const { return infinityStatus == MinusINF; }

  bool operator==(const VRPValue &rhs) const {
    return infinityStatus == rhs.infinityStatus && value == rhs.value;
  }
  bool operator!=(const VRPValue &rhs) const {
    return !operator==(rhs);
  }

  llvm::Optional<bool> cmpLT(const VRPValue &rhs) const {
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

    return value.cmpLT(rhs.value);
  }

  llvm::Optional<bool> cmpLE(const VRPValue &rhs) const {
    return cmpLT(rhs).map([this, &rhs](bool lt) { return lt || *this == rhs; });
  };

  llvm::Optional<bool> cmpGT(const VRPValue &rhs) const {
    return cmpLE(rhs).map([](bool le) { return !le; });
  };
  llvm::Optional<bool> cmpGE(const VRPValue &rhs) const {
    return cmpGT(rhs).map([this, &rhs](bool gt) { return gt || *this == rhs; });
  };

  static llvm::Optional<VRPValue> min(const VRPValue &lhs,
                                      const VRPValue &rhs) {
    return lhs.cmpLE(rhs).map([&lhs, &rhs](bool le) { return le ? lhs : rhs; });
  }

  static llvm::Optional<VRPValue> max(const VRPValue &lhs,
                                      const VRPValue &rhs) {
    return lhs.cmpLE(rhs).map([&lhs, &rhs](bool le) { return le ? rhs : lhs; });
  }

  void print(raw_ostream &os) const;
  void dump() const;
};

template <typename T>
using VRPRange = std::pair<VRPValue<T>, VRPValue<T>>;

// mlir::LatticeElement for the data-flow analysis.
template <typename T>
class VRPLatticeEl {
  using VT = VRPValue<T>;
  VRPRange<T> range;

public:
  bool validate() const {
    return range.first.cmpLE(range.second).getValueOr(false);
  };
  VRPLatticeEl(VRPRange<T> r) : range(r) { assert(validate()); }
  VRPLatticeEl()
      : range(std::make_pair(VT::getMInfinity(), VT::getPInfinity())) {
    assert(validate());
  }
  VRPRange<T> getRange() const { return range; }

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

  void print(raw_ostream &os) const;
  void dump() const;
};

template <typename VRV>
struct VRPAnalysisBase : public ForwardDataFlowAnalysis<VRPLatticeEl<VRV>> {
  VRPAnalysisBase(mlir::MLIRContext *c)
      : ForwardDataFlowAnalysis<VRPLatticeEl<VRV>>(c) {}
  ~VRPAnalysisBase() override = default;

  // Similar to Operation::fold, a client analysis must implement
  // a fold over VRPRange, describing the range of output
  // values, given the ranges for each input operand.
  // failure() can be returned if nothing useful could be computed.
  virtual LogicalResult rangeFold(Operation *op,
                                  ArrayRef<VRPRange<VRV>> operands,
                                  SmallVectorImpl<VRPRange<VRV>> &results) = 0;
  virtual void print(Operation *topLevelOp, raw_ostream &os);
  void dump() const;

protected:
  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<VRPLatticeEl<VRV>> *> operands) final;

private:
  DenseSet<Operation *> visited;
};

// mlir::Attribute with a comparison operator.
struct VRPAttribute : public Attribute {
  VRPAttribute(Attribute a) : Attribute(a) {}
  llvm::Optional<bool> cmpLT(const Attribute &rhs) const;
};

/// Clients of this analysis can either use VRPAnalysis or extend
///   VRPAttribute: To support comparison for more Attribute types.
///   VRPAnalysis: To support more Operations.
struct VRPAnalysis : public VRPAnalysisBase<VRPAttribute> {
  VRPAnalysis(mlir::MLIRContext *c) : VRPAnalysisBase<VRPAttribute>(c) {}
  LogicalResult
  rangeFold(Operation *op, ArrayRef<VRPRange<VRPAttribute>> operands,
            SmallVectorImpl<VRPRange<VRPAttribute>> &results) override;
};

} // namespace mlir
