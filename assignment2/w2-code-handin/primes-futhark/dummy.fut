-- from Helpercode LH2
let mkFlagArray 't [m] 
            (aoa_shp: [m]i64) (zero: t)   --aoa_shp=[0,3,1,0,4,2,0]
            (aoa_val: [m]t  ) : []t   =   --aoa_val=[1,1,1,1,1,1,1]
  let shp_rot = map (\i->if i==0 then 0   --shp_rot=[0,0,3,1,0,4,2]
                         else aoa_shp[i-1]
                    ) (iota m)
  let shp_scn = scan (+) 0 shp_rot       --shp_scn=[0,0,3,4,4,8,10]
  let aoa_len = if m == 0 then 0         --aoa_len= 10
                else shp_scn[m-1]+aoa_shp[m-1]
  let shp_ind = map2 (\shp ind ->        --shp_ind= 
                       if shp==0 then -1 --  [-1,0,3,-1,4,8,-1]
                       else ind          --scatter
                     ) aoa_shp shp_scn   --   [0,0,0,0,0,0,0,0,0,0]
  in scatter (replicate aoa_len zero)    --   [-1,0,3,-1,4,8,-1]
             shp_ind aoa_val             --   [1,1,1,1,1,1,1]
                                     -- res = [1,0,0,1,1,0,0,0,1,0] 


-- from Helpercode LH2
let segmented_scan [n] 't (op: t -> t -> t) (ne: t)
                          (flags: [n]bool) (arr: [n]t) : [n]t =
  let (_, res) = unzip <|
    scan (\(x_flag,x) (y_flag,y) ->
             let fl = x_flag || y_flag
             let vl = if y_flag then y else op x y
             in  (fl, vl)
         ) (false, ne) (zip flags arr)
  in  res


-- taken from https://futhark-lang.org/examples/exclusive-scan.html
let exscan [n] 't (f: t -> t -> t) (ne: t) (xs: [n]t) : [n]t =
  map2 (\i x -> if i == 0 then ne else x)
       (indices xs)
       (rotate (-1) (scan f ne xs))

let nested_inner (mm1: i64, p: i64) : []i64 = 
    let a = replicate mm1 0 :> []i64
    let iot = scan (+) 0 a
    let twom = map (+2) iot  
    let rp = replicate mm1 p :> []i64
    in map (\(j,p) -> j*p) (zip twom rp) 

let flattened_inner (flat_size: i64, p: i64) : []i64 = 
    let aoa_shp = replicate flat_size 1 -- we want have a shape array [flat_size, flat_size, ...]
      let aoa_val = replicate flat_size false
      let flag_array = mkFlagArray aoa_shp false aoa_val :> [flat_size]bool
      let arr = replicate flat_size 1i64  :> [flat_size]i64
      let iots = segmented_scan (+) 0i64 flag_array arr
      -- PART:  let twom = map (+2) iot
      --we apply rule 2 that a map (map f) becomes map f on a flattened array
      let twoms = map (+2) iots 

      -- PART: let rp = replicate mm1 p -- F rule 3
      -- we use rule 3 that states that map (replicate) can be 
      -- flattened to a composition of scan and scatter
      
      let inds = exscan (+) 0 mult_lens
      let size = (last inds) + (last mult_lens)
      let flag = scatter (replicate size 0) inds mult_lens 

      let bool_flag = map  (\ f -> if f != 0        
              then true
              else false
      ) flag 
      let vals = scatter (replicate size 0) inds sq_primes 
      let rp_s = segmented_scan (+) 0i64 bool_flag vals 


      let cast_twoms = twoms 
      let not_primes = map (\(j,p) -> j*p) (zip cast_twoms rp_s)  
      in not_primes



let main (n : i64) : ([]i64, []i64) = 
    let p = 2
    let a = nested_inner (n,  p)
    in (a, a) 