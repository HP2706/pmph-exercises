-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1, -2i32, -2i32, 2i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    3i32
-- }
-- compiled input {
--     [1i32, 0i32, 3i32, 3i32, 3i32, 3i32, 6i32, 7i32, 8i32, 9i32, 10i32]
-- }
-- output {
--     4i32
-- }
-- compiled input {
--     [1i32, 0i32, 1i32, 0i32, 1i32, 0i32, 1i32, 0i32, 1i32, 1i32]
-- }
-- output {
--     2i32
-- }

-- ==
-- entry: onlybench
-- input @ data.in
-- output @ data_same.out

import "lssp"
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
  --in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs

--TODO: implement for a larger dataset to be sure of correctness



-- ==
-- entry: onlybench
-- input @ data.in
-- output @ data_same.out
entry onlybench (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
  --in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs