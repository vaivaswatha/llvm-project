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
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

template <typename VRV>
ChangeResult VRPAnalysis<VRV>::visitOperation(
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

    LatticeElement<VRPLatticeEl<VRV>> foldResult(
        (VRPLatticeEl<VRV>(foldResults[i])));
    result |= lattice.join(foldResult);
  }
  return result;
}

LogicalResult
FloatRangeAnalysis::rangeFold(Operation *op, ArrayRef<VRPRange<float>> operands,
                                SmallVectorImpl<VRPRange<float>> &results) {
  llvm::errs() << "Analyzing operation: ";
  op->print(llvm::errs());
  llvm::errs() << "\n";

  if (isa<math::AbsOp>(op)) {
    llvm::errs() << op->getResult(0).getType() << "\n";
    llvm::errs() << FloatAttr::get(FloatType::getF32(op->getContext()), 0.11);
  }
  return failure();
}

void FloatRangeAnalysis::runOnOperation(Operation *op) {
  FloatRangeAnalysis analysis(op->getContext());
  analysis.run(op);
}

} // namespace mlir