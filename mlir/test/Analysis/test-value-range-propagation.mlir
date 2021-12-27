// RUN: mlir-opt %s -mlir-disable-threading -allow-unregistered-dialect -test-print-value-range-propagation 2>&1 | FileCheck %s

func @abs1(%f: f32) -> f32 {
  %g = math.abs %f : f32
  // CHECK: %0 = math.abs %arg0 : f32 : [0.000000e+00 : f32 ; INF]
  return %g : f32
}

func @abs2(%arg0 : i1) -> f32 {
  cond_br %arg0, ^bb0, ^bb1

^bb0:
  %f = arith.constant -1.0 : f32
  // CHECK: %cst = arith.constant -1.000000e+00 : f32 : [-1.000000e+00 : f32 ; -1.000000e+00 : f32]
  br ^bb2 (%f : f32)

^bb1:
  %g = arith.constant 2.0 : f32
  // CHECK: %cst_0 = arith.constant 2.000000e+00 : f32 : [2.000000e+00 : f32 ; 2.000000e+00 : f32]
  br ^bb2 (%g : f32)

^bb2 (%arg1 : f32):
  %h = math.abs %arg1 : f32
  // CHECK: %1 = math.abs %0 : f32 : [0.000000e+00 : f32 ; 2.000000e+00 : f32]
  return %h : f32
}
