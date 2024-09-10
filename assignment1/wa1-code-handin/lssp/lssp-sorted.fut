-- Parallel Longest Satisfying Segment
-- Small datasets
-- ==
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }  
-- output { 
--    9
-- }
-- compiled input {
--     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
-- }
-- output {
--     10
-- }
-- compiled input {
--     [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
-- }
-- output {
--     1
-- }

-- Benchmarking
-- compiled random input { [100000]i32 }




import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs



-- ==
-- entry: onlybench
-- input @ data.in
-- output @ data_same.out

entry onlybench (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs