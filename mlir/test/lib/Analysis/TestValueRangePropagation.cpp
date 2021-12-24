//===- TestValueRangeAnalysis.cpp - Test value range analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for value range analysis
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/ValueRangePropagation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestValueRangePropagationPass
    : public PassWrapper<TestValueRangePropagationPass, FunctionPass> {
  StringRef getArgument() const final {
    return "test-print-value-range-propagation";
  }
  StringRef getDescription() const final {
    return "Print the results of value range propagation.";
  }
  void runOnFunction() override {
    llvm::errs() << "Testing function: " << getFunction().getName() << "\n";
    VRPAnalysis analysis(&getContext());
    analysis.run(getOperation());
    analysis.print(getOperation(), llvm::errs());
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestValueRangePropagationPass() {
  PassRegistration<TestValueRangePropagationPass>();
}
} // namespace test
} // namespace mlir
