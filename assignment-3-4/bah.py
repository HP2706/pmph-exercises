

""" 
\begin{lstlisting}[language=c]
        float accum, tmpA;
        for (int i = 0; i < N; i++) { // outer loop
            accum = 0;
            for (int j = 0; j < 64; j++) { // inner loop
                tmpA = A[i, j];
                accum = accum  + tmpA*tmpA; // 
                B[i,j] = accum;
            }
    }
    \end{lstlisting}
"""




import numpy as np
N = 2
a = np.array([np.arange(64) for _ in range(N)])

new_a = np.zeros((N,64), dtype=np.int64)

for i in range(N):
    accum = 0
    for j in range(64):
        tmpA = a[i,j]
        accum = accum + tmpA*tmpA
        new_a[i,j] = accum

print(new_a)


