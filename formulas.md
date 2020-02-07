## 2-2-1 Neural Network formulas

![2-2-1 Ann figure](/home/fbgencer/bxb_ann/ann_221.png)



Hidden -> Output
$$
O^{3}_{1} =\sigma(O^{2}_{1}w^{2}_{11}+O^{2}_{2}w^{2}_{21}) 
	=
\sigma(
\begin{bmatrix}
	w^{2}_{11} \\ w^{2}_{21}	
\end{bmatrix}^{T}

\begin{bmatrix}
	O^{2}_{1} \\O^{2}_{2}	
\end{bmatrix}

)
$$
Input -> Hidden
$$
O^{2}_{1} =\sigma(O^{1}_{1}w^{1}_{11}+O^{1}_{2}w^{1}_{21}) \\
O^{2}_{2} =\sigma(O^{1}_{1}w^{1}_{12}+O^{1}_{2}w^{1}_{22}) \\

\begin{bmatrix}
	O^{2}_{1} \\O^{2}_{2}	
\end{bmatrix}
 = \sigma(
\begin{bmatrix}
	w^{1}_{11} && w^{1}_{12} \\ w^{1}_{21} && w^{1}_{22}	
\end{bmatrix}^{T}
\begin{bmatrix}
	O^{1}_{1} \\ O^{1}_{2}	
\end{bmatrix}
 ) \\
\begin{bmatrix}
	O^{1}_{1} \\ O^{1}_{2}	
\end{bmatrix}
 = \begin{bmatrix}
	I_{1} \\ I_{2}	
\end{bmatrix}
$$


### Back propagation

$$
\frac{\partial O^{3}_{1}}{\partial w^{2}_{11}} = O^{2}_{1}\sigma'(I_{O^{3}_{1}}) \\
\frac{\partial O^{3}_{1}}{\partial w^{2}_{21}} = O^{2}_{2}\sigma'(I_{O^{3}_{1}})\\

\frac{\partial O^{3}_{j}}{\partial w^{2}_{ij}} = O^{2}_{i}\sigma'(I_{O^{3}_{j}})\\

%\Delta \begin{bmatrix}
%w^{2}_{11}\\w^{2}_{21}
%\end{bmatrix}
%= 
%\begin{bmatrix}
%\Delta_{O^{3}_{1}}O^{2}_{1}\sigma'(I_{O^{3}_{1}})  \\
%\Delta_{O^{3}_{1}}O^{2}_{2}\sigma'(I_{O^{3}_{1}})
%\end{bmatrix} \\
%=
%\Delta_{O^{3}_{1}}
%\sigma'(I_{O^{3}_{1}})
%\begin{bmatrix}
%O^{2}_{1} \\ O^{2}_{2} 
%\end{bmatrix}
$$


$$
\frac{\partial O^{3}_{1}}{\partial w^{1}_{11}} = w^{2}_{11} O^{1}_{1} \sigma'(I_{O^{2}_{1}})\sigma'(I_{O^{3}_{1}}) \\

\frac{\partial O^{3}_{1}}{\partial w^{1}_{12}} = w^{2}_{21} O^{1}_{1} \sigma'(I_{O^{2}_{2}})\sigma'(I_{O^{3}_{1}})\\

\frac{\partial O^{3}_{1}}{\partial w^{1}_{21}} = w^{2}_{11} O^{1}_{2} \sigma'(I_{O^{2}_{1}})\sigma'(I_{O^{3}_{1}}) \\

\frac{\partial O^{3}_{1}}{\partial w^{1}_{22}} = w^{2}_{21} O^{1}_{2} \sigma'(I_{O^{2}_{2}})\sigma'(I_{O^{3}_{1}})\\

\frac{\partial O^{3}_{k}}{\partial w^{1}_{ij}}  = w^{2}_{jk} O^{1}_{i} \sigma'(I_{O^{2}_{j}})\sigma'(I_{O^{3}_{k}})\\

%\Delta \begin{bmatrix}
%w^{1}_{11}&w^{1}_{12}\\w^{1}_{21} & w^{1}_{22}
%\end{bmatrix}
%= 
%\begin{bmatrix}
%w^{2}_{11} O^{1}_{1} \sigma'(I_{O^{2}_{1}})\sigma'(I_{O^{3}_{1}}) & 
%w^{2}_{21} O^{1}_{1} \sigma'(I_{O^{2}_{2}})\sigma'(I_{O^{3}_{1}}) \\
%w^{2}_{11} O^{1}_{2} \sigma'(I_{O^{2}_{1}}) \sigma'(I_{O^{3}_{1}}) & 
%w^{2}_{21} O^{1}_{2} \sigma'(I_{O^{2}_{2}}) \sigma'(I_{O^{3}_{1}})
%\end{bmatrix} \\
%= 
%\sigma'(I_{O^{3}_{1}})
%\begin{bmatrix}
%w^{2}_{11} \\ w^{2}_{21} 
%\end{bmatrix}^{T}
%\begin{bmatrix}
%O^{1}_{1} \\ O^{1}_{2} 
%\end{bmatrix}
%\begin{bmatrix}
%\sigma'(I_{O^{2}_{1}}) \\ \sigma'(I_{O^{2}_{2}}) 
%\end{bmatrix}
$$

$$
Error \equiv \delta_{O} \\
	 =\Sigma(target-calculated)^2
	 = \Sigma(\Delta)^2
$$

$$
\delta_{O^{3}_{1}} = (\Delta_{O^{3}_{1}})^2 \\
    \frac{\partial \delta_{O^{3}_{1} } }{\partial w^{2}_{11}} = 
    \frac{\partial \delta_{O^{3}_{1}}}{\partial O^{3}_{1}}  \frac{\partial O^{3}_{1}}{\partial w^{2}_{11}} \\
    \frac{\partial \delta_{O^{3}_{1}}}{\partial O^{3}_{1}} = \frac{\partial (target_{1}-{O^{3}_{1}})^2}{\partial O^{3}_{1}} \\
     = -2 \Delta_{O^{3}_{1}} \\
     \frac{\partial O^{3}_{1}}{\partial w^{2}_{11}} = \frac{\partial (\sigma(O^{2}_{1}w^{2}_{11}+O^{2}_{2}w^{2}_{21})) }{\partial w^{2}_{11}} = \frac{\partial (\sigma(I_{O^{3}_{1}})) }{\partial w^{2}_{11}} \\
     = O^{2}_{1}\sigma'(I_{O^{3}_{1}}) \\
\frac{\partial O^{3}_{1}}{\partial w^{2}_{21}} = O^{2}_{2}\sigma'(I_{O^{3}_{1}})
$$

From hidden to input

w^{1}_{11}
$$
\frac{\partial \delta_{O^{3}_{1} } }{\partial w^{1}_{11}} = 
    \frac{\partial \delta_{O^{3}_{1}}}{\partial O^{3}_{1}}  \frac{\partial O^{3}_{1}}{\partial w^{1}_{11}} \\
    \frac{\partial O^{3}_{1}}{\partial w^{1}_{11}} = \frac{\partial (\sigma(O^{2}_{1}w^{2}_{11}+O^{2}_{2}w^{2}_{21})) }{\partial w^{1}_{11}} \\
  \frac{\partial O^{3}_{1}}{\partial w^{1}_{11}}  = w^{2}_{11} \frac{\partial O^{2}_{1}}{\partial w^{1}_{11}} \sigma'(I_{O^{3}_{1}})  \\
    \frac{\partial O^{2}_{1}}{\partial w^{1}_{11}} = \frac{\partial (\sigma(O^{1}_{1}w^{1}_{11}+O^{1}_{2}w^{1}_{21})) }{\partial w^{1}_{11}} 
    = O^{1}_{1} \sigma'(I_{O^{2}_{1}})
    \\
\frac{\partial O^{3}_{1}}{\partial w^{1}_{11}} =  w^{2}_{11} O^{1}_{1} \sigma'(I_{O^{2}_{1}}) \sigma'(I_{O^{3}_{1}})
$$
w^{1}_{12}
$$
\frac{\partial \delta_{O^{3}_{1} } }{\partial w^{1}_{12}} = 
    \frac{\partial \delta_{O^{3}_{1}}}{\partial O^{3}_{1}}  \frac{\partial O^{3}_{1}}{\partial w^{1}_{12}} \\
    \frac{\partial O^{3}_{1}}{\partial w^{1}_{12}} = \frac{\partial (\sigma(O^{2}_{1}w^{2}_{11}+O^{2}_{2}w^{2}_{21})) }{\partial w^{1}_{12}} \\
  \frac{\partial O^{3}_{1}}{\partial w^{1}_{12}}  = w^{2}_{21} \frac{\partial O^{2}_{2}}{\partial w^{1}_{12}} \sigma'(I_{O^{3}_{1}})  \\
    \frac{\partial O^{2}_{2}}{\partial w^{1}_{12}} = \frac{\partial (\sigma(O^{1}_{1}w^{1}_{12}+O^{1}_{2}w^{1}_{22})) }{\partial w^{1}_{12}} 
    = O^{1}_{1} \sigma'(I_{O^{2}_{2}})
    \\
\frac{\partial O^{3}_{1}}{\partial w^{1}_{12}} =  w^{2}_{21} O^{1}_{1} \sigma'(I_{O^{2}_{2}}) \sigma'(I_{O^{3}_{1}})
$$


w^{1}_{21}
$$
\frac{\partial \delta_{O^{3}_{1} } }{\partial w^{1}_{21}} = 
    \frac{\partial \delta_{O^{3}_{1}}}{\partial O^{3}_{1}}  \frac{\partial O^{3}_{1}}{\partial w^{1}_{21}} \\
    \frac{\partial O^{3}_{1}}{\partial w^{1}_{21}} = \frac{\partial (\sigma(O^{2}_{1}w^{2}_{11}+O^{2}_{2}w^{2}_{21})) }{\partial w^{1}_{21}} \\
  \frac{\partial O^{3}_{1}}{\partial w^{1}_{21}}  = w^{2}_{11} \frac{\partial O^{2}_{1}}{\partial w^{1}_{11}} \sigma'(I_{O^{3}_{1}})  \\
    \frac{\partial O^{2}_{1}}{\partial w^{1}_{21}} = \frac{\partial (\sigma(O^{1}_{1}w^{1}_{11}+O^{1}_{2}w^{1}_{21})) }{\partial w^{1}_{21}} 
    = O^{1}_{2} \sigma'(I_{O^{2}_{1}})
    \\
\frac{\partial O^{3}_{1}}{\partial w^{1}_{21}} =  w^{2}_{11} O^{1}_{2} \sigma'(I_{O^{2}_{1}}) \sigma'(I_{O^{3}_{1}})
$$
w^{1}_{22}
$$
\frac{\partial \delta_{O^{3}_{1} } }{\partial w^{1}_{22}} = 
    \frac{\partial \delta_{O^{3}_{1}}}{\partial O^{3}_{1}}  \frac{\partial O^{3}_{1}}{\partial w^{1}_{22}} \\
    \frac{\partial O^{3}_{1}}{\partial w^{1}_{22}} = \frac{\partial (\sigma(O^{2}_{1}w^{2}_{11}+O^{2}_{2}w^{2}_{21})) }{\partial w^{1}_{22}} \\
  \frac{\partial O^{3}_{1}}{\partial w^{1}_{12}}  = w^{2}_{21} \frac{\partial O^{2}_{2}}{\partial w^{1}_{22}} \sigma'(I_{O^{3}_{1}})  \\
    \frac{\partial O^{2}_{2}}{\partial w^{1}_{22}} = \frac{\partial (\sigma(O^{1}_{1}w^{1}_{12}+O^{1}_{2}w^{1}_{22})) }{\partial w^{1}_{22}} 
    = O^{1}_{2} \sigma'(I_{O^{2}_{2}})
    \\
\frac{\partial O^{3}_{1}}{\partial w^{1}_{22}} =  w^{2}_{21} O^{1}_{2} \sigma'(I_{O^{2}_{2}}) \sigma'(I_{O^{3}_{1}})
$$

$$
\frac{\partial O^{3}_{1}}{\partial w^{2}_{11}} = O^{2}_{1}\sigma'(I_{O^{3}_{1}}) \\
\frac{\partial O^{3}_{1}}{\partial w^{2}_{21}} = O^{2}_{2}\sigma'(I_{O^{3}_{1}})\\
\frac{\partial O^{3}_{1}}{\partial w^{1}_{11}} =  w^{2}_{11} O^{1}_{1} \sigma'(I_{O^{2}_{1}}) \sigma'(I_{O^{3}_{1}}) \\
\frac{\partial O^{3}_{1}}{\partial w^{1}_{12}} =  w^{2}_{21} O^{1}_{1} \sigma'(I_{O^{2}_{2}}) \sigma'(I_{O^{3}_{1}}) \\
\frac{\partial O^{3}_{1}}{\partial w^{1}_{21}} =  w^{2}_{11} O^{1}_{2} \sigma'(I_{O^{2}_{1}}) \sigma'(I_{O^{3}_{1}}) \\
\frac{\partial O^{3}_{1}}{\partial w^{1}_{22}} =  w^{2}_{21} O^{1}_{2} \sigma'(I_{O^{2}_{2}}) \sigma'(I_{O^{3}_{1}})
$$
