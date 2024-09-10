
-- entry: main
-- input {
--   [0i64, 1i64, 0i64, 3i64]
--   [2.0f32, -1.0f32, -1.0f32, 2.0f32]
--   [1i64, 2i64, 3i64, 4i64]
--   [1.0f32, 2.0f32, 3.0f32, 4.0f32]
-- }
-- output {
--   [2f32, -2f32, -1f32, 8f32]
-- }
let main[n][m]
    (mat_inds: [n]i64)
      (mat_vals: [n]f32)
      (shp: [m]i64)
      (vct: []f32)
        : [n]f32 =

  let mat_vals = zip mat_inds mat_vals
  let products = map (\(ind, value) -> 
    value * vct[ind]
  ) mat_vals

  in products
