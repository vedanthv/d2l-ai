### Summary

This section explored some of the underpinnings of generalization in machine learning. Some of these ideas become complicated and counterintuitive when we get to deeper models, there, models are capable of overfitting data badly, and the relevant notions of complexity can be both implicit and counterintuitive (e.g., larger architectures with more parameters generalizing better). We leave you with a few rules of thumb:

- Use validation sets (or K-fold cross-validation) for model selection

- More complex models often require more data

- Relevant notions of complexity include both the number of parameters and the range of values that they are allowed to take

- Keeping all else equal, more data almost always leads to better generalization

- This entire talk of generalization is all predicated on the IID assumption. If we relax this assumption, allowing for distributions to shift between the train and testing periods, then we cannot say anything about generalization absent a further (perhaps milder) assumption.

