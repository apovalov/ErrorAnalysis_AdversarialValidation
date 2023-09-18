# ErrorAnalysis and AdversarialValidation

When we count a metric on crossvalidation, we aggregate the errors from all the deferred sample objects into a single number, the metric. For example, we count MSE, MAE, MAPE, or, in classification, LogLoss. A point aggregated metric does not give an understanding of how the error is distributed over different objects, what its distribution is, especially what needs to be changed in the collected ML system to correct the error. The metric can only give a hint that things are bad or good, but it does not tell you what it is because of.

Error analysis is the exact opposite process: we take an aggregated metric and break it down into its components. This gives us an indication of where the system is doing well or poorly.

## Residuals



1. Residuals 
2. Residuals desrubution (hetero.., normal, unbias)
3. Fairness of destribution
4. Use a machine learning model to analyze a machine learning model by learning an advanced adversarial validation technique in the context of residual analysis.




