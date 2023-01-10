
# Policies assessment

## General principles

The simulations are based on the following principles:

1. All instruments apply from 2012 (except the carbon tax which applies from 2014) and are maintained until 2050; they
   are therefore taken into account in the calibration of the model;
2. The different instruments are fully cumulative with each other;
3. The instruments work at full capacity; in particular, a household that carries out a renovation
4. The instruments work at full capacity; in particular, a household that carries out a renovation receives all the aid
   for which it is eligible;
5. The instruments apply equally to private and social housing[^housing];
6. Incentives are perfectly passed on to households, without being captured by energy efficiency vendors in the form of
   an increase in the base price; this mechanism is based on an assumption of perfect competition in the energy
   renovation sector (Nauleau et al., 2015). These assumptions are relatively optimistic. As detailed in the paragraphs
   below, however, policy-specific parameterization elements moderate this optimism. More generally, the study aims at
   least as much to qualitatively illustrate the mechanisms by which the different instruments operate as to provide a
   precise quantitative assessment of their impact. Finally, it is important to remember that integrated energy-economy
   modeling aims to understand mechanisms and quantify orders of magnitude rather than to provide precise results - as
   summarized by the formula modeling for insights, not numbers advocated by the Energy Modeling Forum (Huntington et
   al., 1982). In particular, the inaccuracies of the modeling, i.e. the discrepancies that may exist between
   simulations and observations, can be instructive in the sense that they highlight gaps in the knowledge of the
   processes, thus making it possible to orientate the research priorities.

[^housing]: This hypothesis is particularly optimistic, since in practice the social housing stock is only concerned by
the CEE and the VATr. However, it has little influence, as shown by the results presented below through the low
sensitivity of the renovation of social housing to the level of aid.

## Evaluation Indicators

Version 3.0 of Res-IRF allows for a multi-criteria evaluation of policies. The effectiveness of an intervention -
instrument or mix of instruments - is assessed as its ability to achieve the objectives assigned to it - in this case,
the five objectives defined bellow. To this indicator, which was the core of the evaluation conducted in 2011 for the
CGDD, is now added the efficiency of an intervention, assessed as its capacity to achieve a certain objective at the
lowest cost. This notion only applies to incentive instruments and therefore not to thermal regulations. Two efficiency
indicators are calculated: the **cost effectiveness** and the **leverage effect**. Finally, the differentiation of
households by income category allows us to evaluate the distributional effects.

### Effectiveness

Evaluating the additional effect of an intervention requires the formulation of a counterfactual scenario, without the
intervention under consideration. One of the advantages of the Res-IRF modeling architecture is the simplicity with
which an intervention can be added or removed, facilitating the construction of a counterfactual scenario. An important
difficulty, however, is that the multiplicity of instruments represented opens the door to a multiplicity of
counterfactual scenarios. Specifically, within a bundle of n instruments (at a given parameterization), each instrument
can be evaluated alone, or in interaction with 1 to n - 1 instruments; with n = 6, it is thus possible to simulate 64
scenarios. In the exercise presented here, four scenarios receive particular attention:

- Trend scenario with all policies (AP): includes all instruments in their default variant. This is the scenario on
  which the model is calibrated at the starting year (2012).
- Counterfactual scenario without any policies (ZP for "zero policies"): all instruments are removed [^removed], without
  recalibrating the model. The initial energy consumptions are therefore not the initial energy consumptions are
  therefore not reproduced exactly.
- Trend scenario + (AP+): includes all instruments in their "+" variant [^variant].
- AP scenario without landlord-tenant dilemma (DPL): the behavior of landlords is modelled on that of owner-occupiers ("
  full green value" scenario, {numref}`investment_horizon_2012`). This variant aims to quantify the energy savings that
  would be generated by the different instruments if they were perfectly targeted at the rental stock [^stock].

[^removed]: Except for the reduced VAT rate, which is replaced by the standard 10% VAT rate.
[^variant]: This scenario leads, at the beginning of the period, to a few situations where households occupying housing
classified as G or F receive a total subsidy of more than 100%.
[^stock]: This implies an increased promotion of the various aids to landlords and a specific accompaniment of the work.

Among the 64 possible instrument combinations, the additional effect of each policy can be evaluated in 32 different
ways, depending on the counterfactual situation considered. Two counterfactual situations, obtained by two different
methods, are of particular interest because they make it possible to limit the impact of each instrument:

- AP-1 method: the AP scenario is compared to an alternative scenario without the instrument considered. The difference
  between the two scenarios gives the impact of the instrument in interaction with all other instruments.
- ZP+1" method: the ZP scenario is compared to an alternative scenario with the instrument considered. The difference
  between the two scenarios gives the pure impact of the instrument.

For each instrument, the comparison of the impacts obtained by the two methods makes it possible to evaluate the
importance of its interactions with all the other instruments. Nevertheless, for a better readability of the results, in
the rest of the documentation the estimates in AP-1 are preferred, which correspond to the counterfactual situation that
is closest to reality, in the sense that it is based on a minimum of assumptions.

### Efficiency

The efficiency of the incentive instruments is evaluated here through the indicators of cost-effectiveness and leverage.
To estimate the marginal effect of the instrument at a year 𝑡, scenarios with and without the instrument are compared
at year 𝑡. Since this method is computationally intensive, we limit ourselves to the cut-off points 2015, 2025 and
2035.

#### Cost-Effectiveness (CE)

The cost-effectiveness indicator relates the costs of the incentive to its effectiveness measured in terms of energy
savings. The cost considered corresponds to the tax expenditure in the case of subsidies and the tax revenue in the case
of the tax (i.e., negative cost). The indicator is calculated here in conventional and real energy. The "conventional
energy" metric makes it possible to avoid the heterogeneous behavioral effects between households, which vary greatly
from one year to the next with short-term fluctuations in energy prices. Compared to the "real energy" metric, it leads
to an overestimation of the effectiveness of subsidies by ignoring the rebound effect they generate, and to an
underestimation of the effectiveness of energy taxes, which on the contrary induce a sobriety effect. The two metrics
should produce similar results for WHO, a hybrid instrument. To calculate the cost-effectiveness indicator $CE$, the
energy savings $\Delta E_{t}$ between the scenarios with the incentive present or absent at year $t$ (but in both cases
present until year $t-1$) are compared, applying a discount factor DF:

$$CE_{t} = \frac{\text{Incitation}_{t}}{\Delta E_{t} DF}$$

Using a discount rate of 4% and a lifetime of 26 years, which corresponds to the average of the operations carried out
in the framework of the WHO on the residential building perimeter, the factor $DT$ is taken to be equal to 16.6.

#### Leverage Effect (LE)

The leverage effect LE relates the effectiveness of the instrument, measured in terms of capital expenditure, to the
cost of the incentive. A leverage effect equal to 1 implies that one euro of public money (grant expenditure or tax
revenue) induces an additional investment of one euro. The leverage effect synthesizes several effects, and its value
will depend on the relative share of "additional" and "non-additional" participants:

- The incentive benefits "additional" participants, who would not have invested without it. For these individuals, the
  leverage effect is much greater than 1, since the subsidy is literally the trigger for investment. For example, a 10%
  grant results in a leverage effect equal to 10 - a €100 investment is triggered by a €10 grant.
- The incentive also benefits (and, in our modeling, fully benefits) "non-additional" ("infra-marginal" in economic
  terms) participants, who would have invested without the incentive. While this effect is usually referred to as a "
  dead weight loss", it is not a pure loss in our model as participants adjust their investment choices towards more
  costly options in response to the incentive. Because non-additional participants have heterogeneous preferences, the
  leverage effect may be greater than one for some and close to zero for others.

The formula used applies to all participants and relates the cost of the incentive to the surplus of investment
$\Delta Inv_{t}$ induced by the policy, measured as the difference between two scenarios with the incentive absent or
present at year $t$, but in both cases present until year $t - 1$:

$$EL_{t} = \frac{\Delta Inv_{t}}{\text{Incitation}_{t}}$$


### Fuel poverty

Fuel poverty is measured according to the energy effort rate (EER) indicator that has been a reference in Europe in
recent years (Hills, 2012, p.30), counting the number of households that spend more than 10% of their income on energy
expenses for heating, measured in relation to conventional energy consumption, given in France by the 3CL method of the
DPE. This indicator covers 2.7 million households in 2012. For comparison, the Observatoire de la précarité
énergétique (ONPE, 2016) counts 2.8 million households according to a similar indicator but applied to actual energy
consumption and restricted to the first three deciles of the income distribution. In the present exercise, we favor the
conventional consumption indicator, on the grounds that the actual consumption indicator would not account for certain
restrictions in heating behavior that nevertheless constitute a form of fuel poverty. In complement, one can also
examine how the intensity of heating system use, which reflects indoor comfort, varies with income category. This
indicator can be interpreted as an indirect measure of the ONPE's subjective cold indicator (FR).