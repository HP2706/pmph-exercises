-- Parallel Longest Satisfying Segment

import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
  --in  lssp_seq pred1 pred2 xs
  in  lssp_seq pred1 pred2 xs

--TODO: implement for a larger dataset to be sure of correctness