// RUN: mlir-opt %s -allow-unregistered-dialect -test-print-value-range-propagation 2>&1 | FileCheck %s

func @abs(%f: f32) -> f32 {
  %g = math.abs %f : f32
  return %g : f32
}
