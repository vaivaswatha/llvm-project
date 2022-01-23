//===- ValueRangePropagation.cpp - Value Range Analysis -------------------===//
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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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

llvm::Optional<bool> VRPImplBase::cmpLT(const Attribute &lhs,
                                        const Attribute &rhs) {
  Type lhsType = lhs.getType();
  Type rhsType = rhs.getType();

  if (lhsType != rhsType) {
    return {};
  }
  if (lhsType.isa<FloatType>()) {
    return {lhs.cast<FloatAttr>().getValue() <
            rhs.cast<FloatAttr>().getValue()};
  }
  if (IntegerType lhsIntType = lhsType.dyn_cast<IntegerType>()) {
    if (lhsIntType.isSigned()) {
      return lhs.cast<IntegerAttr>().getValue().slt(
          rhs.cast<IntegerAttr>().getValue());
    }
    // TODO: What to do for signless?
    return lhs.cast<IntegerAttr>().getValue().ult(
        rhs.cast<IntegerAttr>().getValue());
  }
  // Don't know
  return {};
}

using VRPV = VRPAttribute<VRPImplBase>;
using VRPR = VRPRange<VRPImplBase>;

// Given a range for each input operand, try to compute a range for the outputs.
// Partial reference: https://en.wikipedia.org/wiki/Interval_arithmetic
LogicalResult VRPImplBase::rangeFold(Operation *op, ArrayRef<VRPR> operands,
                                     SmallVectorImpl<VRPR> &results) {

  assert(op->getNumOperands() == operands.size());

  if (isa<arith::ConstantOp>(op)) {
    results.push_back(VRPR(op->getAttr("value"), op->getAttr("value")));
    return success();
  }

  if (isa<math::AbsOp>(op)) {
    Type resType = op->getResultTypes()[0];
    FloatAttr zeroFA = FloatAttr::get(resType, 0.0);
    VRPV zeroVal(zeroFA);
    auto abs = [&zeroFA, &resType](const VRPV &v) -> VRPV {
      if (v.isInfinity()) {
        return VRPV::getPInfinity();
      }
      FloatAttr vAttr = v.getValue().cast<FloatAttr>();
      if (vAttr.getValue() < zeroFA.getValue()) {
        APFloat vVal = vAttr.getValue();
        vVal.changeSign();
        return VRPV(FloatAttr::get(resType, vVal));
      }
      return v;
    };
    VRPV lbVal = operands[0].first;
    VRPV ubVal = operands[0].second;
    // In below operations, we assume that VRPV comparisons return Some result.
    assert(*lbVal.cmpLE(ubVal));
    if (*ubVal.cmpLE(zeroVal)) {
      // ubVal <= 0.
      // Take abs() and switch the bounds.
      results.push_back(VRPR(abs(ubVal), abs(lbVal)));
      return success();
    }
    if (*lbVal.cmpLE(zeroVal) && *ubVal.cmpGE(zeroVal)) {
      // lbVal <= 0 <= ubVal.
      // Result is 0 <= max(abs(lbVal), ubVal).
      results.push_back(VRPR(zeroFA, *VRPV::max(abs(lbVal), ubVal)));
      return success();
    }
    // 0 <= lbVal <= ubVal.
    results.push_back(VRPR(lbVal, ubVal));
    return success();
  }

  if (isa<arith::AddFOp>(op) || isa<arith::AddIOp>(op)) {
    // [lb1; ub1] + [lb2; ub2] = [lb1+lb2; ub1+ub2]
    VRPV lb1Val = operands[0].first;
    VRPV ub1Val = operands[0].second;
    VRPV lb2Val = operands[1].first;
    VRPV ub2Val = operands[1].second;
    Type resType = op->getResultTypes()[0];
    auto attrAdd = [&resType](VRPV opd1, VRPV opd2) {
      assert(resType.isa<FloatType>() || resType.isa<IntegerType>());
      return resType.isa<FloatType>()
                 ? VRPV(FloatAttr::get(
                       resType,
                       opd1.getValue().cast<FloatAttr>().getValue() +
                           opd2.getValue().cast<FloatAttr>().getValue()))
                 : VRPV(IntegerAttr::get(
                       resType,
                       opd1.getValue().cast<IntegerAttr>().getValue() +
                           opd2.getValue().cast<IntegerAttr>().getValue()));
    };
    VRPV lbResVal = (lb1Val.isInfinity() || lb2Val.isInfinity())
                        ? VRPV::getMInfinity()
                        : attrAdd(lb1Val, lb2Val);
    VRPV ubResVal = (ub1Val.isInfinity() || ub2Val.isInfinity())
                        ? VRPV::getPInfinity()
                        : attrAdd(ub1Val, ub2Val);
    results.push_back(VRPR(lbResVal, ubResVal));
    return success();
  }

  if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
    // [lb1; ub1] * [lb2; ub2] =
    // [
    //   min(lb1*lb2, lb1*ub2, ub1*lb2, ub1*ub2);
    //   max(lb1*lb2, lb1*ub2, ub1*lb2, ub1*ub2)
    // ]
    VRPV lb1Val = operands[0].first;
    VRPV ub1Val = operands[0].second;
    VRPV lb2Val = operands[1].first;
    VRPV ub2Val = operands[1].second;
    Type resType = op->getResultTypes()[0];
    auto attrMul = [&resType](VRPV opd1, VRPV opd2) {
      assert(resType.isa<FloatType>() || resType.isa<IntegerType>());
      return resType.isa<FloatType>()
                 ? VRPV(FloatAttr::get(
                       resType,
                       opd1.getValue().cast<FloatAttr>().getValue() *
                           opd2.getValue().cast<FloatAttr>().getValue()))
                 : VRPV(IntegerAttr::get(
                       resType,
                       opd1.getValue().cast<IntegerAttr>().getValue() *
                           opd2.getValue().cast<IntegerAttr>().getValue()));
    };
    if (lb1Val.isInfinity() || lb2Val.isInfinity() || ub1Val.isInfinity() ||
        ub2Val.isInfinity()) {
      results.push_back(VRPR(VRPV::getMInfinity(), VRPV::getPInfinity()));
      return success();
    }
    auto min4 = [](VRPV opd1, VRPV opd2, VRPV opd3, VRPV opd4) {
      VRPV min1 = *opd1.cmpLT(opd2) ? opd1 : opd2;
      VRPV min2 = *opd3.cmpLT(opd4) ? opd3 : opd4;
      return *min1.cmpLT(min2) ? min1 : min2;
    };
    auto max4 = [](VRPV opd1, VRPV opd2, VRPV opd3, VRPV opd4) {
      VRPV min1 = *opd1.cmpLT(opd2) ? opd2 : opd1;
      VRPV min2 = *opd3.cmpLT(opd4) ? opd4 : opd3;
      return *min1.cmpLT(min2) ? min2 : min1;
    };
    VRPV prods[] = {attrMul(lb1Val, lb2Val), attrMul(lb1Val, ub2Val),
                    attrMul(ub1Val, lb2Val), attrMul(ub1Val, ub2Val)};
    results.push_back(VRPR(min4(prods[0], prods[1], prods[2], prods[3]),
                           max4(prods[0], prods[1], prods[2], prods[3])));
    return success();
  }

  return failure();
}

} // namespace mlir