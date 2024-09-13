-- mkFlagArray: inspired by lecture notes page 48
-- mkFlagArray: inspired by lecture notes page 48
let mkFlagArray 't [m]  
  (aoa_shp: [m]i64) 
  (zero: t) 
  (aoa_val: [m]t) : []t = 

  let shp_rot = map (\i -> 
    if i == 0 then 0
    else aoa_shp[i-1]
  ) (iota m)
  
  let shp_scn = scan (+) 0 shp_rot
  let aoa_len = if m == 0 then 0 
                else shp_scn[m-1] + aoa_shp[m-1]

  let shp_ind = map2 (\shp ind -> 
    if shp == 0 then -1
    else ind
  ) aoa_shp shp_scn

  in scatter (replicate aoa_len zero) shp_ind aoa_val



let main : []i64 = 
    let mat_shp_flags = [1, 2, 1, 1, 5, 1, 1, 1, 8, 1]
    let aoa_val = replicate 10 1i64
    let flags = mkFlagArray mat_shp_flags 0i64 aoa_val
    in flags