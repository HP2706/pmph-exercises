-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }

-- output @ ref10000000.out

type tuple = (i64, i64)

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

let primesFlat (n : i64) : []i64 =
  let sq_primes   = [2i64, 3i64, 5i64, 7i64]
  let len  = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes
      let flat_size = reduce (+) 0 mult_lens




      --------------------------------------------------------------
      -- The current iteration knows the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the correct `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`. 
      
    -- LECTURE SLIDES:
    --let nested = map (\p ->
      --let m = n ‘div‘ p in -- distribute map
      --let mm1 = m - 1 in -- distribute map
      --let iot = iota mm1 in -- F rule 4
    --  let twom= map (+2) iot in -- F rule 2
    --  let rp = replicate mm1 p in -- F rule 3
    --  in map (\(j,p) -> j*p) (zip twom rp) -- F rule 2
    --  ) sqrt_primes

      -- the thing we are truing to flatten
      --let composite = map (\mm1 ->
        --let iot = scan (+) 0 replicate flat_size 0 --  apply rule 4
        --let twom = map (+2) iot -- F rule 2
        --let rp = replicate mm1 p -- F rule 3
        --in map (\(j,p) -> j*p) (zip twom rp) -- F rule 2
      --) mult_lens

      -- PART: let iot = scan (+) 0 replicate flat_size 0 
      --rule 4 map (scan op) becomes segmented_scan (op)
      
      let aoa_shp = replicate flat_size 1 -- we want have a shape array [flat_size, flat_size, ...]
      let aoa_val = replicate flat_size false
      let flag_array = mkFlagArray aoa_shp false aoa_val :> [flat_size]bool
      let arr = replicate flat_size 1i64 :> [flat_size]i64
      let iots = segmented_scan (+) 0i64 flag_array arr 


      -- PART:  let twom = map (+2) iot
      --we apply rule 2 that a map (map f) becomes map f on a flattened array
      let twoms = map (+2) iots 

      -- PART: let rp = replicate mm1 p -- F rule 3
      -- we use rule 3 that states that map (replicate) can be 
      -- flattened to a composition of scan and scatter      
      let inds = scan (+) 0 mult_lens
      let size = (last inds) + (last mult_lens)
      let flag = scatter (replicate size 0) inds mult_lens 

      let bool_flag = map  (\ f -> if f != 0        
              then true
              else false
      ) flag 
      let vals = scatter (replicate size 0) inds sq_primes 
      let rp_s = segmented_scan (+) 0i64 bool_flag vals :> [flat_size]i64


      let cast_twoms = twoms 
      let not_primes = map (\(j,p) -> j*p) (zip cast_twoms rp_s)  


      --let not_primes = replicate flat_size 0i8
      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

       let zero_array = replicate flat_size 0i8
       let mostly_ones= map (\ x -> if x > 1 then 1i8 else 0i8) (iota (len+1))
       let prime_flags= scatter mostly_ones not_primes zero_array
       let sq_primes = filter (\i-> (i > 1i64) && (i <= n) && (prime_flags[i] > 0i8))
                              (0...len)

       in  (sq_primes, len)

  in sq_primes

-- RUN a big test with:
--   $ futhark cuda primes-flat.fut
--   $ echo "10000000i64" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
-- or simply use futhark bench, i.e.,
--   $ futhark bench --backend=cuda primes-flat.fut
let main (n : i64) : []i64 = primesFlat n
