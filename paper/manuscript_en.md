# Clinical Factors of Recurrent Diabetic Ketoacidosis: An Analysis of a Russian Cohort Using Statistics and Interpretable Machine Learning

Oleg Sadykhov (1), Vasiliy Leonenko (1)

(1) Artificial Intelligence Technologies Faculty, ITMO University,
49 Kronverksky Pr., St. Petersburg, 197101, Russia

Corresponding author: Oleg Sadykhov, ovelsad@niuitmo.ru, ORCID 0009-0008-9062-7523
Second author: Vasiliy Leonenko, vnleonenko@itmo.ru, ORCID 0000-0001-7070-6584

<!--
STATUS. English translation of manuscript_ru.md (current version: diagnostic
framing, no "core" wording, top-k intersection language). When editing content,
edit all three files: manuscript_ru.md, manuscript_en.md, manuscript_en.tex.

OUTSTANDING (from authors):
- ethics approval: committee name, protocol number, date (placeholder in
  Declarations);
- clinical interpretation of factors in Discussion (deferred to physicians).
-->

## Abstract

**Purpose.** To identify clinical factors associated with a recurrent course of
diabetic ketoacidosis (DKA) in a Russian cohort and to assess, using
interpretable machine learning, how far these groups differ in clinical profile
at admission.

**Methods.** A retrospective single-center study, predictors and outcome refer
to the same hospitalization. Time to event was not recorded, so the model is
diagnostic rather than prognostic. The outcome was established in 273 of 289
patients (recurrent 87, single 186). Groups were compared under a fixed
statistical protocol with Benjamini-Hochberg correction. Independently, four
interpretable models were fitted on 25 outcome-independent features without
multicollinearity, the split preceding preprocessing. Feature contributions were
estimated by SHAP and compared with the statistics. Two hypotheses were
pre-specified, on alcohol intake and complex models outperforming logistic
regression.

**Results.** Ten features were significantly associated with the outcome.
Discrimination was moderate, ROC-AUC 0.71-0.78 by nested cross-validation and
0.72-0.80 on the held-out test. Five features entered the top-10 by SHAP in all
four models, four statistically significant and the fifth (glucose at admission)
found only multivariately. Under bootstrap, two of the five remained shared
(89-92% of replicates). The clinical hypothesis was confirmed, odds ratio 2.27
(95% CI 1.23-4.20). The methodological hypothesis was not, DeLong found no
significant differences between models.

**Conclusion.** The clinical profile separates the groups moderately, and the
upper parts of the factor lists from two independent methodologies coincide.

**Keywords:** diabetic ketoacidosis, recurrent course, risk factors,
interpretable machine learning, SHAP, diagnostic model

## 1 Introduction

Diabetic ketoacidosis is a life-threatening complication of diabetes mellitus,
characterized by hyperglycemia, metabolic acidosis and an elevated concentration
of ketone bodies [1]. Between 2003 and 2014 the United States recorded 1,760,101
primary DKA hospitalizations, and the number of discharges with a principal
diagnosis rose from 118,808 in 2003 to 188,965 in 2014 [2]. After the acute
state is resolved, single cases may progress to repeated episodes, and a
recurrent course carries substantially higher mortality [3, 4].

Repeated DKA episodes are rarely explained by a single cause. Risk factors for
repeated admissions include young age, psychiatric disorders, infections,
non-adherence to therapy, alcohol and psychoactive substance abuse, and
socioeconomic circumstances [5, 6, 7]. The set of factors is heterogeneous,
results across studies agree poorly, and for the Russian population the factors
of a recurrent course have not been described. Despite decades of clinical
experience, optimal DKA management strategies remain a subject of debate [8].

Closest to our task is a case-control observational study that matched patients
with recurrent and single courses on age, diabetes duration, sex and ethnicity
[9]. After matching, few appreciable differences between the groups remained, and
the main trigger of an episode in both groups was missed insulin injections. With
40 patient pairs such a result is hard to interpret unambiguously, it may mean
either an absence of differences or a lack of power. The question of which
patient characteristics are associated with a recurrent course remains open.

Machine learning allows the same task to be viewed from another angle. Published
DKA models reach ROC-AUC of about 0.82-0.85 [10, 11], but they address a
different task, they predict the first episode or the fact of hospitalization
within a set time window, they rely on large foreign electronic health record
databases with a high proportion of missing values, and on other samples their
metrics usually do not reproduce [11]. It has also been shown [12, 13] that many
models work as a black box, their conclusions are hard to interpret, which
reduces clinicians' trust and becomes a barrier to adoption. Current work in
diabetology applies prediction explanation methods, primarily SHAP and LIME
[14, 15], but for a recurrent course of DKA such tools have hardly been used.

In this work we identify the clinical factors associated with a recurrent course
of DKA in a Russian cohort and assess how far a patient's clinical profile at
admission can distinguish a recurrent course from a single episode. The key
feature of the approach is that we obtain the set of factors along two
independent paths, classical statistics under an agreed protocol and SHAP on top
of interpretable machine learning models, after which we compare the results.
Such a comparison shows to what extent the set of factors depends on the chosen
analytical approach, which matters especially in a small sample.

We test two pre-specified hypotheses, the first of them clinical, concerning
alcohol intake, which the literature describes as a factor of a recurrent course
alongside psychoactive substance use, with excess alcohol named among the
frequent triggers of an episode [7]. The association is also supported by studies
of repeated admissions [9, 16]. What remains is to find out whether it carries
over to a Russian sample. The methodological hypothesis concerns model choice, a
systematic review of clinical prediction models found no advantage of machine
learning over logistic regression [17], yet this has not been tested on the task
of recurrent DKA. The answer has a practical consequence, if complexity yields no
gain, a simple and interpretable model is preferable.

## 2 Methods

### 2.1 Design and Data Source

A retrospective single-center study on de-identified data with cross-sectional
outcome ascertainment, both predictors and outcome refer to the same DKA
hospitalization, with no observation period between them. The analysis included
patients hospitalized with diabetic ketoacidosis between 2019 and 2025.
Inclusion criteria were age 18 years or older and a confirmed diagnosis of DKA.
The only exclusion criterion was an undetermined outcome. The initial sample
comprised 289 patients, and the outcome was established in 273.

### 2.2 Outcome

The target variable was a recurrent course of DKA (0 for a single episode, 1 for
a recurrent course). A patient was assigned to the recurrent group if more than
one DKA episode had been recorded in total. The episode count combined the number
of the patient's prior visits to this hospital with the same diagnosis from the
medical records, the history at admission when the patient had not been observed
here before, and the current hospitalization. A single course was defined as a
first and only presentation.

The outcome was ascertained retrospectively, so at the time the predictors were
measured it had in most cases already occurred. The data contain no fixed
observation window, no dates of individual episodes and no time to event, and
there is no censoring. Under the TRIPOD classification the model is diagnostic,
it recognizes a state that has already occurred rather than predicting a future
event. It does not admit direct comparison with models predicting
hospitalization within 30, 90 or 180 days. The task addressed is the separation
of a recurrent from a single course by the patient's clinical profile and the
identification of factors associated with it. The consequences for
interpretation are discussed in the limitations section.

### 2.3 Hypotheses

Both hypotheses were formulated in advance, before data analysis, on the basis
of the literature.

**Clinical hypothesis.** H0: alcohol intake in the day before an episode is not
associated with a recurrent course (odds ratio equals 1). H1: an association
exists (odds ratio differs from 1). Test, the protocol test for a 2x2 table,
effect measure, odds ratio with a 95% confidence interval, with the
Benjamini-Hochberg correction for multiple comparisons.

**Methodological hypothesis.** H0: complex models (random forest, gradient
boosting) and logistic regression do not differ in discrimination. H1: complex
models outperform logistic regression by ROC-AUC. Test, the paired DeLong test
for correlated ROC curves on the same patients, separately on the out-of-fold
training part and on the held-out test set, with the Benjamini-Hochberg
correction.

We put forward no other hypotheses. All other statistical comparisons are
descriptive, and features with significant differences between groups are treated
as an exploratory result requiring confirmation on an independent sample.

### 2.4 Predictors

We used indicators collected at hospitalization, demographic data (age, sex),
history and therapy of diabetes (type, duration, age at onset, insulin delivery
method, daily dose), laboratory indicators at admission (blood gases,
electrolytes, creatinine, urea, glucose, HbA1c, lipid panel), complications
(chronic kidney disease by stage and albuminuria, diabetic retinopathy and
polyneuropathy) and a lifestyle factor (alcohol intake).

The models were trained on a single feature set, 25 indicators without marked
multicollinearity. Age is almost linearly related to age at onset and diabetes
duration (Pearson correlation about 0.99), so age at onset was removed as
linearly dependent. The composition of the set was determined only by the mutual
correlation of predictors and did not depend on the outcome, so feature
selection introduces no bias into the quality estimate.

### 2.5 Sample Size

The sample size was not calculated in advance, the study is retrospective, so all
available observations entered the analysis, 273 patients with an established
outcome (recurrent course proportion 31.9%, 87 events). With 87 events for 25
features there are few observations of the rare class, which raises the risk of
overfitting, so we limited dimensionality and applied regularization and class
weighting. Signs of overfitting were sought in three ways, by comparing estimates
on nested cross-validation and the held-out test, by the out-of-bag estimate of
the random forest, and by 95% confidence intervals of the metrics from bootstrap.

### 2.6 Missing Data

The proportion of missing values per indicator ranged from a few percent to 58%
(HbA1c, sodium, potassium, lactate). We assumed the values were missing at random
conditional on the observed variables, and accounted for their panel structure,
where laboratory indicators go missing in groups. We compared imputation
strategies, median and mode, k-nearest neighbors, and multiple imputation by
chained equations. The imputer was trained only on the training part within
folds, and on the test set missing values were filled from the nearest neighbors
of the training part, without using the outcome or other test observations.

The working choice was k-nearest neighbors imputation with five neighbors. It
outperformed the other strategies in model quality and did not appreciably
distort the shape of the distributions, when comparing distributions before and
after imputation by the Kolmogorov-Smirnov test no significant discrepancies were
found. This check is indicative, since the compared samples are nested within one
another, and it cannot be read as proof that the distributions coincide. The
number of neighbors was compared separately (k from 3 to 11) by out-of-fold
ROC-AUC on the training part, and k = 5 proved best.

### 2.7 Statistical Analysis

Normality was tested by the Shapiro-Wilk test for n below 50 and by the
Kolmogorov-Smirnov test in the Lilliefors variant for n of 50 or more.
Quantitative indicators with a normal distribution were described by mean and
standard deviation with a 95% confidence interval, and when departing from
normality by median and quartiles. Groups were compared by the Student t-test,
the Mann-Whitney U test or the Brunner-Munzel test depending on normality and
equality of variances (Levene test). Proportions in 2x2 tables were compared by
the Pearson chi-square test when the minimum expected frequency exceeded 10 and
by the Fisher exact test when the expected frequency was below 10. The confidence
interval for proportions was computed by the Clopper-Pearson method, and the
effect size by the odds ratio with a 95% confidence interval. For multiple
comparisons the Benjamini-Hochberg correction was applied. Areas under the ROC
curves of two models on the same patients were compared by the DeLong test for
correlated curves.

### 2.8 Model Development

The sample was split once, stratified, in an 80 to 20 ratio, training part 218
patients, test 55. The split was performed before any preprocessing, with a fixed
random seed. All preprocessing (imputation, scaling, encoding of categorical
features, balancing) was performed inside the pipeline and trained only on the
training folds.

The choice of strategies and model families was made entirely on the training
part, the held-out test was not consulted at this stage. We compared seven
families (logistic regression, random forest, LightGBM, CatBoost, XGBoost,
support vector machine, k-nearest neighbors) along the axes of imputation,
balancing, categorical encoding and transformation of quantitative features, the
comparison criterion being out-of-fold ROC-AUC on the training part.

Four families representing different model classes were taken forward, logistic
regression as a linear model and the reference for the methodological hypothesis,
random forest as a bagging method, and XGBoost and CatBoost as gradient boosting
variants, the latter handling categorical features natively. The support vector
machine and k-nearest neighbors lagged appreciably (out-of-fold ROC-AUC 0.677 and
0.631 against 0.713-0.780 for the others) and were not considered further.
LightGBM was excluded as redundant with XGBoost, both implement gradient boosting
over trees, and the difference between them (0.724 against 0.716) is small
against the spread of estimates across folds.

Class imbalance was handled by class weighting, and synthetic methods (SMOTE,
SMOTENC, BorderlineSMOTE, ADASYN) were compared separately. Hyperparameters were
tuned by Bayesian search in nested cross-validation, the procedure quality was
assessed by an outer five-fold cross-validation, while within each training fold
the hyperparameters were tuned by a separate three-fold cross-validation.

### 2.9 Performance Assessment

The main metrics were chosen to be threshold-independent, ROC-AUC and PR-AUC with
95% confidence intervals from bootstrap (2000 replicates). These answer the
question of how far the clinical profile separates the groups and do not depend on
an arbitrary threshold choice.

Probability calibration was checked by out-of-fold calibration curves and the
Brier score, raw probabilities were compared with the Platt and isotonic
regression methods, the application rule was set in advance, calibration is used
only where it lowers the Brier score by at least 0.005.

In addition, for a clear view of the error structure, we report the confusion
matrix. The cut-off point was chosen by a neutral criterion, the maximum of the
Youden index on the out-of-fold probabilities of the training part, then fixed
and applied to the held-out test, which was used once. This point does not serve
as a clinical recommendation, since the outcome is ascertained retrospectively
there is no screening task here, and the cut-off serves only to illustrate the
ratio of type I errors (false positives) to type II errors (false negatives).

### 2.10 Interpretation

Feature contributions were estimated by SHAP for all four models, LinearSHAP for
logistic regression, TreeSHAP for random forest and XGBoost, and the built-in
SHAP value computation for CatBoost. The contributions of one-hot encoded
categories were summed back to the source feature, and feature importance was
measured by the mean absolute SHAP value.

Feature importance was examined at two levels, answering different questions.

The first level is agreement among the models. We checked whether the ranking of
features coincides across models or depends on the choice of algorithm. We
assessed this by the pairwise Spearman rank correlation between the full rankings,
and by the intersection of the top lists, for which at thresholds of top-5, top-10
and top-15 we counted how many features enter the top by SHAP in all four models
at once.

The dependence of this intersection on the sample was checked by bootstrap, 200
replicates, in each the training part was resampled with replacement, all four
models were refitted with fixed hyperparameters, and SHAP was computed on the
observations not drawn into the sample.

The second level is the comparison of machine learning with classical statistics.
We checked whether the features shared across the top-10 of all models coincide
with the list of indicators significant in the between-group comparison. The two
lists were obtained independently, the results of the statistical comparison did
not affect the set of features fed to the models. In addition, for each indicator
from Table 1 we counted in how many of the four models it entered the top-10 by
SHAP.

### 2.11 Software and Reporting

The analysis was performed in Python 3.11 with fixed library versions, and the
code is open. Reporting follows the STROBE recommendations for observational
studies, and for the part concerning model development the TRIPOD+AI
recommendations were additionally taken into account.

## 3 Results

### 3.1 Participant Characteristics

Of the initial cohort of 289 patients, 16 without an established outcome were
excluded. The analytical sample comprised 273 patients, 87 with a recurrent
course (31.9%) and 186 with a single episode. The stratified split gave a
training part of 218 patients (69 with a recurrent course, 149 with a single
episode) and a test part of 55 (18 and 37 respectively).

Table 1 collects the indicators with a significant between-group difference after
the Benjamini-Hochberg correction, 10 in total.

**Table 1.** Features with a significant between-group difference
(Benjamini-Hochberg correction, q below 0.05)

| Indicator | Single (n=186) | Recurrent (n=87) | q | OR (95% CI) |
|---|---|---|---|---|
| Daily insulin dose, units | 34 [0; 45] | 44 [35; 54] | <0.0001 | - |
| No insulin therapy, % | 28.9 | 1.1 | <0.0001 | - |
| HDL cholesterol, mmol/L | 1.09 [0.83; 1.38] | 1.26 [1.06; 1.58] | 0.0065 | - |
| Diabetes duration, years | 6 [0; 12.5] | 9 [4; 15] | 0.0139 | - |
| HbA1c, % | 12.56 [10.28; 13.64] | 10.90 [8.91; 12.66] | 0.0196 | - |
| No retinopathy, % | 65.4 | 48.1 | 0.0196 | - |
| No kidney damage (G0), % | 56.3 | 25.8 | 0.0196 | - |
| No albuminuria (A0), % | 52.7 | 27.8 | 0.0196 | - |
| Alcohol intake, % | 15.3 | 29.1 | 0.0234 | 2.27 [1.23; 4.20] |
| Polyneuropathy, % | 52.1 | 69.0 | 0.0281 | 2.05 [1.17; 3.57] |

Quantitative indicators are given as median [Q1; Q3]. For categorical indicators
the proportion of the stated level among known values in the group is given,
missing values are not included in the denominator. The odds ratio is computed
only for binary indicators.

For the insulin delivery method, the main difference is driven not by the
proportion on pump therapy but by the proportion of patients on no insulin
therapy, 28.9% in the single-episode group against 1.1% in the recurrent group.
This reflects a structural signal of disease onset, since a first episode by
definition cannot be assigned to a recurrent course.

### 3.2 Clinical Hypothesis: Alcohol Intake

The feature value is known for 269 of 273 patients. Among patients with a
recurrent course, 29.1% (25 of 86) had consumed alcohol in the day before the
episode, against 15.3% (28 of 183) in the single-episode group. The minimum
expected frequency in the 2x2 table equals 16.9, so by protocol the Pearson
chi-square test was applied, statistic 7.01 at one degree of freedom, p = 0.008.
We reject the null hypothesis. The odds ratio was 2.27 (95% CI 1.23-4.20), that
is, alcohol intake the day before roughly doubles the odds of a recurrent course.
After the Benjamini-Hochberg correction the association retains significance
(q = 0.023). The direction of the effect agrees with the literature [7, 9, 16].

### 3.3 Discrimination

**Table 2.** Discrimination by nested cross-validation

| Model | ROC-AUC | SD |
|---|---|---|
| Random forest | 0.775 | 0.019 |
| XGBoost | 0.748 | 0.038 |
| CatBoost | 0.735 | 0.050 |
| Logistic regression | 0.713 | 0.029 |

SD is the standard deviation across the outer folds of nested cross-validation.

**Table 3.** Discrimination on the held-out test

| Model | ROC-AUC (95% CI) |
|---|---|
| Logistic regression | 0.802 [0.672; 0.912] |
| XGBoost | 0.769 [0.630; 0.888] |
| CatBoost | 0.739 [0.587; 0.868] |
| Random forest | 0.724 [0.576; 0.851] |

Confidence intervals were obtained by bootstrap (2000 replicates).

Test estimates lie in the range 0.72-0.80, and nested cross-validation estimates
in the range 0.71-0.78. The confidence intervals on the test are wide (0.26 in
width on average) and overlap strongly, so the models cannot be ranked by the
test result alone. Moreover the ranking reversed relative to nested
cross-validation, the random forest, the leader on cross-validation, came last on
the test, while logistic regression, last on cross-validation, gave the best
result on the test. With 55 patients in the test part such a discrepancy is
expected and itself serves as a warning against choosing a model by a single
held-out sample.

PR-AUC on the training part holds in the range 0.52-0.55 against a random
classifier baseline of about 0.32.

As a sign of overfitting we considered a systematic excess of the estimates
obtained on the training part over the estimate on the held-out test beyond the
95% confidence intervals. There is no such excess, for all four models the nested
cross-validation estimate falls within the confidence interval of the test
estimate. In addition, for the random forest two resampling schemes on the
training part agree, the out-of-bag estimate 0.749 [0.683; 0.815] and the
out-of-fold estimate 0.772 [0.710; 0.833] mutually cover each other with their
confidence intervals. Neither check proves the absence of overfitting, but they
show that the discrepancies between estimates do not exceed the statistical
uncertainty at this sample size.

Pairwise comparison of the four models by the DeLong test with the
Benjamini-Hochberg correction over 6 comparisons revealed no significant pair
either on the out-of-fold training part or on the held-out test. The smallest
values were obtained when comparing logistic regression with the complex models on
the training part (p after correction 0.178). This does not prove the equality of
the models, since at this sample size the power of the test is low, but neither
does it give grounds to single out any model as the best.

### 3.4 Methodological Hypothesis

The pre-specified methodological hypothesis was tested separately (Table 4). On
the training part all three complex models are nominally above logistic
regression (difference 0.04-0.05), but after correction the difference is not
significant for any. On the held-out test the sign of the difference is reversed,
logistic regression is above all three complex models by 0.03-0.08, and here too
the differences are not significant.

**Table 4.** Complex models versus logistic regression, DeLong test

| Sample | Complex model | AUC complex | AUC logreg | p (BH) |
|---|---|---|---|---|
| OOF | Random forest | 0.772 | 0.719 | 0.09 |
| OOF | XGBoost | 0.761 | 0.719 | 0.09 |
| OOF | CatBoost | 0.764 | 0.719 | 0.09 |
| Test | Random forest | 0.724 | 0.802 | 0.50 |
| Test | XGBoost | 0.769 | 0.802 | 0.55 |
| Test | CatBoost | 0.739 | 0.802 | 0.50 |

OOF is out-of-fold predictions on the training part. The Benjamini-Hochberg
correction is applied separately within each sample, over three comparisons.

The null hypothesis could not be rejected, the superiority of complex models over
logistic regression was not demonstrated on this sample. We stress the correct
reading, this is not proof of the equivalence of the models, since with 218
patients on the training part and 55 on the test the power of the test is limited.

### 3.5 Calibration and Error Structure

Raw probabilities turned out to be better calibrated than any of the recomputed
variants. The Brier score on out-of-fold predictions was 0.178-0.189 for raw
probabilities, 0.185-0.192 after the Platt method, and 0.190-0.193 after isotonic
regression. The rule set in advance (to apply calibration only if the Brier score
drops by at least 0.005) was not met for any model, so raw probabilities are used
throughout the further analysis.

We note a side effect, calibration also reduced discrimination (for the random
forest from 0.772 to 0.744 after the Platt method). The sigmoid transformation is
itself monotone and cannot change the order of probabilities, and with it the
ROC-AUC. The drop is produced not by it but by the implementation of the
procedure, the calibration wrapper refits the base model on cross-validation folds
and averages their predictions, so the model itself changes, not only the
probability scale. At this sample size this loss is noticeable.

The error structure on the held-out test at the Youden-index cut-off point is
given in Table 5.

**Table 5.** Confusion matrix on the held-out test, Youden-index cut-off point

| Model | Threshold | TP | FP | FN | TN | Sens. | Spec. |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.315 | 16 | 13 | 2 | 24 | 0.889 | 0.649 |
| CatBoost | 0.250 | 14 | 15 | 4 | 22 | 0.778 | 0.595 |
| XGBoost | 0.380 | 13 | 10 | 5 | 27 | 0.722 | 0.730 |
| Random forest | 0.415 | 11 | 10 | 7 | 27 | 0.611 | 0.730 |

Test part, 18 patients with a recurrent course and 37 with a single episode. FP
is a type I error, FN is a type II error.

The matrix shows what is not visible from the ROC-AUC, at similar discrimination
the models distribute errors between the two types differently. Logistic
regression misses only 2 recurrent cases out of 18 but wrongly assigns 13 of 37
patients to that class. The random forest behaves the opposite way, fewer false
alarms but 7 cases missed out of 18. XGBoost gives the most balanced picture. The
spread underscores the instability of the cut-off point at this sample size, so we
build substantive conclusions on threshold-independent metrics.

### 3.6 Negative Results

The effect of transforming quantitative features (manual clinical transformation
and the Yeo-Johnson transformation against standardization only) was checked for
all four models. There is no gain for any, for logistic regression the discrepancy
between variants is 0.004 by ROC-AUC, for CatBoost 0.005, and for XGBoost the
result coincides to the third decimal across all three variants (0.712). The last
is expected, since decision trees are invariant to monotone transformations of
features, and the logarithm together with the Yeo-Johnson transformation does not
change the order of values, and hence the split points. The only noticeable
discrepancy was given by the random forest (0.773 without transformation, 0.756
under manual, 0.775 under Yeo-Johnson), but even this has nothing to do with the
transformation, the manual transformation variant is assembled from three feature
blocks and changes the column order, while the random forest at a fixed seed
selects random subsets of features by their indices.

Stacking with logistic regression as the meta-model over the out-of-fold
predictions of the base models gave the best combination (random forest and
XGBoost) with ROC-AUC 0.7699, which does not exceed the best single model
(0.772). Adding base models did not improve quality.

Synthetic balancing also gave no gain. ROC-AUC for the synthetic methods was
0.715-0.725 (SMOTENC 0.716, BorderlineSMOTE 0.715, ADASYN 0.722, SMOTE over
one-hot 0.725), whereas without balancing it was 0.729, and with class weighting
0.737. The Brier score meanwhile rises from 0.187 to 0.193-0.200, that is,
calibration worsens. The best result was given by the non-synthetic class
weighting, which was applied in all final models.

### 3.7 Feature Importance

The full feature rankings by mean absolute SHAP value agree moderately between
models (Table 6). As expected, agreement is highest between the random forest and
XGBoost, both of which build tree ensembles. Agreement is lowest between CatBoost
and the rest.

**Table 6.** Pairwise Spearman rank correlation between SHAP rankings

| | CatBoost | Logreg | Forest | XGBoost |
|---|---|---|---|---|
| **CatBoost** | 1.00 | 0.49 | 0.58 | 0.47 |
| **Logreg** | 0.49 | 1.00 | 0.71 | 0.51 |
| **Forest** | 0.58 | 0.71 | 1.00 | 0.91 |
| **XGBoost** | 0.47 | 0.51 | 0.91 | 1.00 |

The moderate agreement of the full rankings is explained by the ends of the
lists, where the absolute SHAP values are small and the ranks unstable. The upper
part of the lists coincides considerably more, and the size of the overlap depends
on the cut-off threshold. In the top-5, two features are shared across all four
models, insulin delivery method and daily insulin dose. In the top-10 there are
five such features, HbA1c, HDL cholesterol and glucose at admission being added.
In the top-15 there are eight. There is no distinguished number here, and the
growth of the overlap as the threshold is relaxed is expected, there are only 25
features, and at a threshold covering the whole list all would be shared. What is
informative, therefore, is not the growth but the composition of the overlap at a
tight threshold, where a chance intersection is unlikely.

The overlap at the top-10 holds unevenly across the sample. Under bootstrap of the
training part (200 replicates) insulin delivery method and daily insulin dose
remain shared across all models in 92% and 89% of replicates, whereas HbA1c, HDL
cholesterol and glucose at admission do so in 32-40%. In individual models these
three features enter the top-10 considerably more often, in 47-72% of replicates.
The gap between the two figures reflects a property of the criterion itself, the
requirement of simultaneous presence in all four models is more sensitive to
perturbation of the data than the ranking of each model separately. The same pair
of features that is stable under resampling is already shared across all models at
the tighter top-5 threshold. Without resampling the picture repeats, when SHAP is
computed on the held-out test instead of the training part the rankings of
individual models change little (Spearman 0.97-1.00), while the overlap at the
top-10 shrinks from five features to four. Sample size also plays a part, at 218
observations the exclusion of a few patients noticeably changes the ranking.

The differences between models fall on features outside this overlap and reflect
their construction. Logistic regression additionally raises electrolytes (sodium
and potassium), the random forest and XGBoost raise diabetes duration and total
cholesterol, and CatBoost raises the type of diabetes.

Comparison with classical statistics shows that of the five features shared across
the top-10 of all models, four are also significant in the between-group
comparison, insulin delivery method, daily insulin dose, HbA1c and HDL
cholesterol. The fifth, glucose at admission, showed no significant between-group
difference in the univariate comparison but entered the top-10 in all four models.

The full picture of correspondence is given in Table 7. Indicators with the
smallest q values enter the top-10 in all models, and as q grows the number of
models declines. The agreement of the two approaches is strongest for the most
significant indicators and gradually weakens.

**Table 7.** Correspondence between significance in the between-group comparison
and SHAP contribution

| Indicator | q | In top-10 SHAP, models of 4 |
|---|---|---|
| Insulin delivery method | <0.0001 | 4 |
| Daily insulin dose | <0.0001 | 4 |
| HDL cholesterol | 0.0065 | 4 |
| HbA1c | 0.0196 | 4 |
| Glucose at admission | not significant | 4 |
| Kidney damage (CKD, G) | 0.0196 | 3 |
| Urea at admission | not significant | 3 |
| Total cholesterol | not significant | 3 |
| Diabetes duration | 0.0139 | 2 |
| Retinopathy | 0.0196 | 2 |
| Albuminuria (CKD, A) | 0.0196 | 2 |
| Polyneuropathy | 0.0281 | 1 |
| Alcohol intake | 0.0234 | 0 |

Shown are indicators significant in the between-group comparison and indicators
that entered the top-10 by SHAP in at least three models. The q value is taken
from Table 1.

The discrepancies in the lower part of the table are expected, since the two
quantities measure different things. The between-group test evaluates an isolated
difference in a single indicator, whereas the mean absolute SHAP is the feature's
contribution to the prediction, averaged over the whole cohort and computed in the
presence of the remaining 24 variables. A binary feature with moderate prevalence
receives a small value under such averaging, even if it is informative for its
subgroup of patients.

## 4 Discussion

We assessed how far a patient's clinical profile at admission can distinguish a
recurrent course of DKA from a single episode, and identified the factors
associated with the outcome. Discrimination holds in the range 0.71-0.78 on
nested cross-validation and 0.72-0.80 on the held-out test.

One of the main results is the features shared by two independent approaches, the
multivariate SHAP analysis and the univariate statistical comparison. Insulin
delivery method, daily insulin dose, HbA1c, HDL cholesterol and glucose at
admission enter the top-10 by SHAP in all four machine learning models, that is,
they do not depend on the choice of algorithm. Four of these five features are
also significant in the between-group comparison under the statistical protocol.
This matters methodologically, classical statistics evaluates each feature in
isolation, whereas SHAP evaluates its contribution within a multivariate model
accounting for the remaining variables, so the approaches rest on different
assumptions.

The approaches do not, however, coincide in everything, and these differences are
of a regular nature. Glucose at admission entered the shared top-10 of the models,
although it was not significant in the univariate between-group comparison, that
is, the multivariate model singled out an indicator that on its own does not
separate the groups. There are also the reverse cases, some significant indicators
enter the top-10 in only some models (Table 7). The statistical test and SHAP
measure different things, so the discrepancy between them carries additional
information rather than indicating an error. It is precisely this cross-comparison
of two independent methodologies that forms the methodological side of the work,
and on a small sample it helps to separate a signal common to the methods from the
peculiarities of each of them.

The clinical hypothesis was confirmed, alcohol intake in the day before an episode
is associated with a recurrent course, odds ratio 2.27 (95% CI 1.23-4.20). This
carries a risk factor known from the literature over to the Russian population
[7, 9, 16]. We note a limitation of interpretation, the design is cross-sectional,
so this concerns an association rather than a proven causation.

The methodological hypothesis was not confirmed. The practical conclusion, at this
sample size added model complexity does not pay off, and logistic regression is
preferable as the simplest and most interpretable, with natural probability
calibration. The random forest and XGBoost give the same quality through nonlinear
interactions, and CatBoost is convenient for its native handling of categorical
features, but none of them brings a gain. It is also telling that the model
ranking on the held-out test came out reversed relative to the ranking on nested
cross-validation, with 55 patients in the test part the differences between models
are indistinguishable from random fluctuation. This negative result agrees with a
systematic review [17], which on the material of other clinical tasks also found
no advantage of machine learning over logistic regression.

A direct comparison of discrimination with published models is incorrect, since
they address a different task. On a cohort of adults with type 1 diabetes a
ROC-AUC of about 0.82 was obtained [11], up to 0.85 on a cohort of children [10],
but both works predict the first episode or hospitalization within a set time
window on large foreign databases.

## 5 Limitations

The main limitation is the design. The outcome was ascertained retrospectively, so
at the time the predictors were measured it had in most cases already occurred.
The model is diagnostic, it does not predict a future event but separates already
formed groups by clinical profile.

Diabetes duration differed between groups, a median of 9 years [4; 15] in the
recurrent course against 6 years [0; 12.5] in the single episode (q = 0.0139).
This difference may reflect not risk but the very definition of the outcome, the
longer the disease lasts, the longer the period over which a repeat episode could
be recorded, and the data contain no observation interval common to all patients.
Building a prognostic model requires a prospective cohort with episode dates,
observation time and censoring, which constitutes a natural continuation of the
work.

The overlap of features across all models is only partially confirmed. Under
bootstrap of the training part, insulin delivery method and daily insulin dose
remain shared across all models in 92% and 89% of replicates, while HbA1c, HDL
cholesterol and glucose at admission do so in 32-40%, so the latter three should
be regarded as requiring confirmation. The frequencies obtained are an upper
bound, the hyperparameters were fixed under resampling, and the variability of
their tuning is not built into the estimate. Resampling from a single cohort,
moreover, says nothing about reproducibility on data from another center.

Separately we note a structural signal in the insulin delivery method feature. The
proportion of patients on no insulin therapy was 28.9% in the single-episode group
against 1.1% in the recurrent group, in our cohort the absence of insulin therapy
at hospitalization almost always corresponds to disease onset, and a first episode
by definition cannot be assigned to a recurrent course. The contribution of this
feature therefore partly reproduces the definition of the outcome, and it cannot
be read as an independent risk factor, although it is precisely this feature that,
more often than the rest, turns out to be shared across all models.

The remaining limitations concern the data. The sample is single-center and small,
with no external validation on an independent population. The proportion of missing
values in part of the laboratory indicators is high, up to 58%, and although
imputation did not appreciably distort the distributions, uncertainty remains. A
number of features named in the literature as leading for a recurrent course
(psychiatric disorders, treatment adherence, socioeconomic circumstances) are
absent from our data, and this limits the attainable discrimination.

## 6 Conclusions

A patient's clinical profile at admission separates a recurrent from a single
course of DKA with moderate ability, ROC-AUC 0.71-0.78 by nested cross-validation
and 0.72-0.80 on the held-out test. Five features enter the top-10 by SHAP in all
four models at once, insulin delivery method, daily insulin dose, HbA1c, HDL
cholesterol and glucose at admission. Four of these five are also significant in
the between-group comparison, that is, they are singled out along two independent
methodological paths. Under bootstrap of the training part two of them remain
shared across all models, insulin delivery method and daily insulin dose (92% and
89% of replicates), the other three in 32-40%.

The clinical hypothesis was confirmed, alcohol intake in the day before an episode
is associated with a recurrent course, odds ratio 2.27 (95% CI 1.23-4.20). The
methodological hypothesis was not confirmed, the superiority of complex models over
logistic regression could not be demonstrated, and stacking and synthetic
balancing likewise brought no advantage, which at this data volume argues in favor
of simple interpretable models.

The model is diagnostic rather than prognostic over a time horizon. A prognostic
model requires a prospective cohort with episode dates and observation time,
together with external validation on data from another center.

## Supplementary Materials

The full result tables (cohort characteristics, comparison of preprocessing
strategies, hyperparameter tuning, calibration, cut-off points and confusion
matrices, pairwise DeLong comparisons, stacking, synthetic balancing, the
frequencies with which features turn out to be shared across all models under
bootstrap) and the SHAP figures are available in the open project repository.

## Acknowledgements

The authors thank the physicians who provided the de-identified clinical data.

## Declarations

**Funding.** The study was carried out without external funding.

**Conflict of interest.** The authors declare no conflict of interest relevant to
the content of this article.

**Ethics approval and consent to participate.** [To be completed after the ethics
committee decision is obtained. Required form: the full committee name, protocol
number and date, together with a statement of compliance with the Declaration of
Helsinki of 1964 and its later amendments. For a retrospective study on
de-identified data, a statement of exemption from the requirement of approval with
the same details is admissible.]

**Consent for publication.** Not applicable, the work was performed on
de-identified data, individual patient information is not published.

**Data availability.** Patient data cannot be published, as this would breach
confidentiality. Access is possible on reasonable request to the corresponding
author within ethical constraints.

**Materials availability.** Not applicable.

**Code availability.** The analysis code is open and available at
https://github.com/ovelsad/dka-recurrence-ml, and includes fixed library
versions and instructions for reproduction.

**Author contributions.** O. Sadykhov: study conception, data preparation and
analysis, statistical analysis, model development and validation, interpretation
of results, preparation of the first draft of the manuscript. V. Leonenko: study
conception and design, scientific supervision, analysis methodology, verification
of results, critical revision and editing of the manuscript. Both authors read and
approved the final version.

## References

1. Benoit, S.R., Zhang, Y., Geiss, L.S., Gregg, E.W., Albright, A.: Trends in
   diabetic ketoacidosis hospitalizations and in-hospital mortality - United
   States, 2000-2014. MMWR Morb. Mortal. Wkly. Rep. 67(12), 362-365 (2018).
   https://doi.org/10.15585/mmwr.mm6712a3

2. Desai, D., Mehta, D., Mathias, P., Menon, G., Schubart, U.K.: Health care
   utilization and burden of diabetic ketoacidosis in the U.S. over the past
   decade: a nationwide analysis. Diabetes Care 41(8), 1631-1638 (2018).
   https://doi.org/10.2337/dc17-1379

3. Santos, S.S., Ramaldes, L.A.L., Dualib, P.M., Gabbay, M.A.L., Sa, J.R.,
   Dib, S.A.: Increased risk of death following recurrent ketoacidosis
   admissions: a Brazilian cohort study of young adults with type 1 diabetes.
   Diabetol. Metab. Syndr. 15(1), 85 (2023).
   https://doi.org/10.1186/s13098-023-01054-5

4. Gibb, F.W., Teoh, W.L., Graham, J., Lockman, K.A.: Risk of death following
   admission to a UK hospital with diabetic ketoacidosis. Diabetologia 59(10),
   2082-2087 (2016). https://doi.org/10.1007/s00125-016-4034-0

5. Mohler, R., Lotharius, K., Moothedan, E., Goguen, J., Bandi, R., Beaton, R.,
   Knecht, M., Mejia, M.C., Khoury, M., Sacca, L.: Factors contributing to
   diabetic ketoacidosis readmission in hospital settings in the United States:
   a scoping review. J. Diabetes Complications 38(10), 108835 (2024).
   https://doi.org/10.1016/j.jdiacomp.2024.108835

6. Dhatariya, K.K., Glaser, N.S., Codner, E., Umpierrez, G.E.: Diabetic
   ketoacidosis. Nat. Rev. Dis. Primers 6(1), 40 (2020).
   https://doi.org/10.1038/s41572-020-0165-1

7. Brandstaetter, E., Bartal, C., Sagy, I., Jotkowitz, A., Barski, L.: Recurrent
   diabetic ketoacidosis. Arch. Endocrinol. Metab. 63(5), 531-535 (2019).
   https://doi.org/10.20945/2359-3997000000158

8. Kanzwl, S.M.O.S., Alhajri, A.H.M., Mohammed, Y.J.A., et al.: A comparative
   effectiveness of intravenous fluids and insulin regimens in the acute
   management of diabetic ketoacidosis (DKA) and hypoglycemia: a systematic
   review. Cureus 17(10), e94902 (2025). https://doi.org/10.7759/cureus.94902

9. Cooper, H., Tekiteki, A., Khanolkar, M., Braatvedt, G.: Risk factors for
   recurrent admissions with diabetic ketoacidosis: a case-control
   observational study. Diabet. Med. 33(4), 523-528 (2016).
   https://doi.org/10.1111/dme.13004

10. Williams, D.D., Ferro, D., Mullaney, C., Skrabonja, L., Barnes, M.S.,
    Patton, S.R., Lockee, B., Tallon, E.M., Vandervelden, C.A., Schweisberger,
    C., Mehta, S., McDonough, R., Lind, M., D'Avolio, L., Clements, M.A.: An
    "all-data-on-hand" deep learning model to predict hospitalization for
    diabetic ketoacidosis in youth with type 1 diabetes: development and
    validation study. JMIR Diabetes 8, e47592 (2023).
    https://doi.org/10.2196/47592

11. Li, L., Lee, C.C., Zhou, F.L., Molony, C., Doder, Z., Zalmover, E., Sharma,
    K., Juhaeri, J., Wu, C.: Performance assessment of different machine
    learning approaches in predicting diabetic ketoacidosis in adults with type
    1 diabetes using electronic health records data. Pharmacoepidemiol. Drug
    Saf. 30(5), 610-618 (2021). https://doi.org/10.1002/pds.5199

12. Rudin, C.: Stop explaining black box machine learning models for high stakes
    decisions and use interpretable models instead. Nat. Mach. Intell. 1(5),
    206-215 (2019). https://doi.org/10.1038/s42256-019-0048-x

13. Stiglic, G., Kocbek, P., Fijacko, N., Zitnik, M., Verbert, K., Cilar, L.:
    Interpretability of machine learning-based prediction models in healthcare.
    WIREs Data Min. Knowl. Discov. 10(5), e1379 (2020).
    https://doi.org/10.1002/widm.1379

14. Ahmed, S., Kaiser, M.S., Hossain, M.S., Andersson, K.: A comparative
    analysis of LIME and SHAP interpreters with explainable ML-based diabetes
    predictions. IEEE Access 13, 37370-37388 (2025).
    https://doi.org/10.1109/ACCESS.2024.3422319

15. Emi-Johnson, O.G., Nkrumah, K.J.: Predicting 30-day hospital readmission in
    patients with diabetes using machine learning on electronic health record
    data. Cureus 17(4), e82437 (2025). https://doi.org/10.7759/cureus.82437

16. Bradford, A.L., Crider, C.C., Xu, X., Naqvi, S.H.: Predictors of recurrent
    hospital admission for patients presenting with diabetic ketoacidosis and
    hyperglycemic hyperosmolar state. J. Clin. Med. Res. 9(1), 35-39 (2017).
    https://doi.org/10.14740/jocmr2792w

17. Christodoulou, E., Ma, J., Collins, G.S., Steyerberg, E.W., Verbakel, J.Y.,
    Van Calster, B.: A systematic review shows no performance benefit of machine
    learning over logistic regression for clinical prediction models. J. Clin.
    Epidemiol. 110, 12-22 (2019). https://doi.org/10.1016/j.jclinepi.2019.02.004
