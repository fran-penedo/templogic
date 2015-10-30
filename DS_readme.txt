Dataset format. 
Three input variables:

data   - Matrix of real numbers that contains the signals. 
         Size: Nsignals x Nsamples
         Each row rapresents a signal, each column corresponds to a 
         sampling time.

t      - Column vector of real numbers containing the sampling times 
         Size: 1 x Nsamples

labels - Column vector of real numbers that contains the labels for the  
         signals in data. 
         Size: 1 x Nsignals
         The label is +1 if the corresponging signal belongs to the 
         positive class C_p
         The label is -1 if the corresponging signal belongs to the 
         negative class C_N
         


