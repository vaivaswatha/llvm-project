//===- ValueRangeAnalysis.cpp - Value Range Analysis ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/ValueRangePropagation.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include <utility>

namespace mlir {

template <typename VRV>
ChangeResult VRPAnalysisBase<VRV>::visitOperation(
    Operation *op, ArrayRef<LatticeElement<VRPLatticeEl<VRV>> *> operands) {

  SmallVector<VRPRange<VRV>> opdsRange(
      llvm::map_range(operands, [](LatticeElement<VRPLatticeEl<VRV>> *value) {
        return value->getValue().getRange();
      }));

  SmallVector<VRPRange<VRV>> foldResults;
  if (failed(rangeFold(op, opdsRange, foldResults))) {
    return this->markAllPessimisticFixpoint(op->getResults());
  }

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  ChangeResult result = ChangeResult::NoChange;
  for (unsigned i = 0, e = foldResults.size(); i != e; ++i) {
    LatticeElement<VRPLatticeEl<VRV>> &lattice =
        this->getLatticeElement(op->getResult(i));

    // The VRP lattice doesn't satisfy chain condition.
    // Need to break it by check if an Operation was previously
    // visited and if it has widened since then.
    // If so, then push it to <-INF, INF>. TODO.

    result |= lattice.join(foldResults[i]);
  }
  return result;
}

template <typename VRV>
void VRPAnalysisBase<VRV>::print(Operation *topLevelOp, raw_ostream &os) {
  topLevelOp->walk([&os, this](mlir::Operation *op) {
    for (Value result : op->getOpResults()) {
      LatticeElement<VRPLatticeEl<VRV>> *lattice =
          this->lookupLatticeElement(result);
      if (lattice && !lattice->isUninitialized()) {
        os << result << " : ";
        lattice->getValue().print(os);
        os << "\n";
      }
    }
  });
}

llvm::Optional<bool> VRPAttribute::cmpLT(const Attribute &rhs) const {
  Type thisType = getType();
  Type rhsType = rhs.getType();

  if (thisType != rhsType) {
    return {};
  }
  if (thisType.isa<FloatType>()) {
    return {cast<FloatAttr>().getValue() < rhs.cast<FloatAttr>().getValue()};
  }
  if (IntegerType thisIntType = thisType.dyn_cast<IntegerType>()) {
    if (thisIntType.isSigned()) {
      return cast<IntegerAttr>().getValue().slt(
          rhs.cast<IntegerAttr>().getValue());
    }
    cast<IntegerAttr>().getValue().ult(rhs.cast<IntegerAttr>().getValue());
  }
  // Don't know
  return {};
}

LogicalResult
VRPAnalysis::rangeFold(Operation *op, ArrayRef<VRPRange<VRPAttribute>> operands,
                       SmallVectorImpl<VRPRange<VRPAttribute>> &results) {

  if (isa<math::AbsOp>(op)) {
    VRPValue<VRPAttribute> lb(FloatAttr::get(op->getResultTypes()[0], 0.0));
    VRPValue<VRPAttribute> ub(VRPValue<VRPAttribute>::getPInfinity());
    VRPRange<VRPAttribute> resRange(lb, ub);
    results.push_back(resRange);
    return success();
  }
  return failure();
}

} // namespace mlir