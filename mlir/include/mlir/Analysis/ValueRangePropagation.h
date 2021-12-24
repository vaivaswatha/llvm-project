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
// The type parameter T must have the following operations defined:
//   bool operator<(const T &lhs, const T &rhs)
//   bool operator==(const T &lhs, const T &rhs)
template <typename T>
class VRPValue {

  enum InfinityStatus {
    NotINF,   // Not infinity
    PlusINF,  // +INF
    MinusINF, // -INF
  } infinityStatus;
  T value;

  VRPValue(InfinityStatus is) : infinityStatus(is){};

public:
  VRPValue(const T &v) : infinityStatus(NotINF), value(v){};
  static VRPValue getPInfinity() { return VRPValue(PlusINF); }
  static VRPValue getMInfinity() { return VRPValue(MinusINF); }

  bool isInfinity() const { return infinityStatus != NotINF; }
  bool isPInfinity() const { return infinityStatus == PlusINF; }
  bool isMInfinity() const { return infinityStatus == MinusINF; }

  bool operator==(const VRPValue &rhs) const {
    return infinityStatus == rhs.infinityStatus && value == rhs.value;
  }
  bool operator<(const VRPValue &rhs) const {
    return (isMInfinity() && !rhs.isMInfinity()) ||
           (!isPInfinity() && rhs.isPInfinity()) ||
           (!isInfinity() && !rhs.isInfinity() && value < rhs.value);
  }

  bool operator<=(const VRPValue &rhs) const {
    return *this < rhs || *this == rhs;
  };
  static const VRPValue &min(const VRPValue &lhs, const VRPValue &rhs) {
    return lhs <= rhs ? lhs : rhs;
  }
  static const VRPValue &max(const VRPValue &lhs, const VRPValue &rhs) {
    return lhs <= rhs ? rhs : lhs;
  }
};

template <typename T>
using VRPRange = std::pair<VRPValue<T>, VRPValue<T>>;

// mlir::LatticeElement for the data-flow analysis.
template <typename T>
class VRPLatticeEl {
  using VT = VRPValue<T>;
  VRPRange<T> range;

public:
  bool validate() const { return range.first <= range.second; };
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
        std::make_pair(VT::min(lhs.range.first, rhs.range.second),
                       VT::max(lhs.range.second, rhs.range.second)));
  }
};

template <typename VRV>
struct VRPAnalysis : public ForwardDataFlowAnalysis<VRPLatticeEl<VRV>> {
  VRPAnalysis(mlir::MLIRContext *c)
      : ForwardDataFlowAnalysis<VRPLatticeEl<VRV>>(c) {}
  ~VRPAnalysis() override = default;

  // Similar to Operation::fold, a client analysis must implement
  // a fold over VRPRange, describing the range of output
  // values, given the ranges for each input operand.
  // failure() can be returned if nothing useful could be computed.
  virtual LogicalResult rangeFold(Operation *op,
                                  ArrayRef<VRPRange<VRV>> operands,
                                  SmallVectorImpl<VRPRange<VRV>> &results) = 0;

protected:
  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<VRPLatticeEl<VRV>> *> operands) final;
};

struct FloatRangeAnalysis : public VRPAnalysis<float> {
  FloatRangeAnalysis(mlir::MLIRContext *c) : VRPAnalysis<float>(c) {}
  LogicalResult rangeFold(Operation *op, ArrayRef<VRPRange<float>> operands,
                          SmallVectorImpl<VRPRange<float>> &results) override;

  static void runOnOperation(Operation *op);
};

} // namespace mlir
