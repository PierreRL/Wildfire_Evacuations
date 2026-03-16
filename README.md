# STAT405 Project

_Pierre Lardet_

In this project, I will explore the causes for evacuation orders from Canadian wildfires. In particular, I seek to ask the question of whether some populations (e.g. indigenous populations) and/or regions are treated differently in terms of whether or not an evacuation order will be 

### Data

- Data on evacuation orders on Wildfires from 1980 to 2019. https://zenodo.org/records/5703323
- I may also link this data with other sources such as the Canadian National Fire Database (https://cwfis.cfs.nrcan.gc.ca/ha/nfdb) which contains more detail on the characteristics of the fires, and the Canadian Census which contains more detail on the communities that the fires are in.

The head of the evacuation order data looks like this:

| EvacID | Year | Province | Location   | Lat       | Long       | Population | PopulationNotes | FN_Reserve | EvacSize             | IssueDate | EndDate   | EvacReason | Region | Group | FireSource | RepDOY | FireIgnit | FireSize    |
|------:|-----:|:---------|:-----------|----------:|-----------:|:-----------|:----------------|:-----------|:---------------------|:----------|:----------|:-----------|:-------|:------|:-----------|------:|:----------|------------:|
| 1 | 1980 | MB | Clear Lake | 50.678605 | -99.911954 | NA | NA | no | Very Small (1 to 100) | 5/23/1980 | 5/27/1980 | Threat | SC | SC3 | NFDB.poly | 142 | Human | 22914.74086 |

### Model

As a simple first model, we can fit a Bayesian regression:

$Y_i \sim \text{Bern}(p_i)\\$
$p_i = \text{logistic}(\alpha + \beta^\top C_i + \gamma^\top F_i)\\$
$\alpha \sim \mathcal{N}(0,\sigma_a^2)\\$
$\beta \sim \mathcal{N}(0,\sigma_b^2)\\$
$\gamma \sim \mathcal{N}(0,\sigma_c^2)$

where $i$ is a wildfire, $Y_i$ is whether or not an evacuation order was sent out, $p_i$ is the probability that the evacuation order is sent out, $C_i$ are data on the community that the fire is in and $F_i$ are characteristics of the fire itself, $\alpha$ is the intercept, $\beta$ are the coefficients for the community characteristics, and $\gamma$ are the coefficients for the fire characteristics. I include priors on the parameters $\alpha$, $\beta$, and $\gamma$ to complete the Bayesian model with the variance parameters $\sigma_a^2$, $\sigma_b^2$, and $\sigma_c^2$ to be determined.

Note that $C_i$ and $F_i$ could include different data types which will need to be processed e.g. into one-hot encodings for categorical variables, and standardization for continuous variables.

Thus, this is a Bayesian logistic regression. More advanced models could include interaction terms between the community and fire characteristics, or could include hierarchical structure to account for the fact that some fires may be in the same region or province.

### Questions

The main scientific questions would seek to interpret the parameters $\beta$ to understand how the community characteristics affect the probability of an evacuation order being sent out, and to interpret the parameters $\gamma$ to understand how the fire characteristics affect the probability of an evacuation order being sent out. In particular, I would be interested in whether certain community characteristics (e.g. indigenous populations) are associated with a lower probability of an evacuation order being sent out, which could indicate potential bias in the decision-making process for evacuation orders.