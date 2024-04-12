---
id: Simple Report
tags:
  - eng3px3
date: "2024-02-05"
title: NVF for affordable housing
---

The high-level net value function for which is defined by performance parameters and conversion factors:

$$
\text{NVF} = \text{HouseSalesRevenue} - \text{LabourCost} - \text{EnergyCost} - \text{MaterialsCost} - \text{R\&D} - \text{UpfrontConstructionCost}
$$

Where the performance parameters and conversion factors are defined by the following:

- **Productivity and Housing construction rate**: This is defined by the rate of construction and the number of prefabricated units sold.
  The conversion factor is considered by the sold units, and such generate revenue.
  The assumption here is that there are specific quota of units to be sold that meet the production capacity, as well as the price per unit sold are within market value.

- **Labour cost**: This is defined by the operational cost for given number of workers to build the houses and automations.
  The conversion factors are derived from the average wage per hour, number of working hours per year, as well as number of workers on the project.
  The assumption here is that workers are paid accordingly to their job and the number of hours they work. (i.e. no overtime, $35/hour, 45 hours/week)

- **Energy cost**: This is defined by the energy consumption for the construction and operation of the houses.
  The conversion factors are derived from the consumption per square foot, total operational area, energy price.
  The assumption here is that the energy efficiency of workplace are within the standard, and using market price for energy consumption per kWh.

- **Materials cost**: This is defined by the cost of materials for the construction of the houses.
  The conversion factors are derived from the average cost of material per square foot, and the total area of the houses, and the number of house built per year.
  The assumption, similar to as above, is that there is a certain quota of houses to be built, cost of raw materials required for construction meet standards and policies from rule makers.

- **R&D**: This is defined by the yearly budget for research and development to innovate and improve both the construction and the prefabricated units.
  The conversion factors is a lump sump of the budget allocated for R&D. The assumption for this is that the budget will be able to afford the best team and resources to innovate on current design.

- **Upfront construction cost**: This is defined by the initial investment to start the operation, including factory setup and equipment purchases, compliance with building codes, and other considerations.
  The conversion factors is an amortized, one-time cost of the initial investment. This will be reflected as the capital expenditure needed for the project.

Some of the following considerations are made for the aforementioned [[thoughts/university/eng-3px3/Net Value Function#Net Value functions|NVF]], as well as the performance parameters and conversion factors:

- **Environmental**: The focus on material uses should be vital, such that all materials are sustainable and non-toxic, to reduce emissions. Additionally, the energy consumption would also be increased due to utilisation of automation software and robotics. Therefore, The NVF should be reflected where both `EnergyCost` and `MaterialsCost` would be increased from initial assumptions and due diligence.

- **Regulatory**: Compliance with building codes and different considerations for prefabricated homes are recognized. Compliance can implies additional costs, and therefore `UpfrontConstructionCost` could increase. This could also affect operational feasibility of the projects if any of regulatory requirements are not met.

- **Ethical and DEI**: Possible **DEI** concerns include the wage gap, the small number of workforce due to robots and automation, and the potential displacement of workers. Additionally, DEI are also taken into account to provide affordable housing to different socio-economics classes, such that it aligns with broader social objectives. However, this would then also increase `UpfrontConstructionCost`, similar to regulatory considerations.
