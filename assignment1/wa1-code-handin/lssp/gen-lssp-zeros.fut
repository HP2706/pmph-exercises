
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
--  in  lssp_seq pred1 pred2 xs
  in  lssp_seq pred1 pred2 xs

