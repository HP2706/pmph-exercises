-- Parallel Longest Satisfying Segment
--
-- Small dataset
--
-- ==
-- entry: main
-- 
-- input { [0i32,0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }
-- output { 10i32 }
--
-- input { [1i32, -2, -2, 0, 0, 0, 3, 4, -6] }
-- output { 3i32 }
--
-- input { [0i32, 1, 0, 0 ,3, 0, 0, 4, 0] }
-- output { 2i32 }
--
-- input { [0i32, 1, 0, 0, 1]}
-- output { 2i32 }

-- Benchmarking
-- compiled random input { [100000]i32 }

import "lssp-seq"
import "lssp"

let main (xs: []i32) : i32 =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs


-- ==
-- entry: onlybench
-- input @ data.in
-- output @ data_zeros.out

entry onlybench (xs: []i32) : i32 =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs



