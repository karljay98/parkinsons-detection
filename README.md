# parkinsons-detection

Abstract
This study contains the findings of using non-invasive devices to detect Parkinson’s disease or not. Machine learning methods eXtreme boosting gradient and Random forests are utilised to identify the accuracy of the models used to identify if an individual has Parkinson’s disease. The data set used was obtain from 42 people and produced over 5000 recording over 26 different attributes. Throughout the paper the accuracy of these machine learning models is assessed and scrutinised based on their strengths and weaknesses. The process of deciding what data to use involved feature selection and removing certain data attributes as they irrelevant in predicting if an individual has Parkinson’s. Once this was done a number of python libraries were used to visualise and analyse the data generated. A literature review discussing the pathology of Parkinson’s disease and highlighting current work being undertaken in the tackling of Parkinson’s.  Findings from this study proved that Machine learning algorithms can be used in the detection of Parkinson’s with a high accuracy value. Thus, providing an opportunity to develop medical devices that are more sensitivity in detecting vocal features and removing irrelevant features such as coughing and noise.
Introduction
The aim of this project is to design a model that will accurately detect the presence of Parkinson’s disease (PD) in an individual. This model will use machine learning (ML) (gradient boost), NumPy, pandas and scikit-learn in detecting if the individual has Parkinson’s disease or not. XGBoost will make use of decision trees that are generated based on the input data; the benefit of using XGBoost is that the accuracy of approximations can be increased due to the use of second-order gradients and advanced regularization. Currently there are no biomarkers screen tests that can be used to reveal if an individual has PD.

Parkinson’s
Parkinson’s disease is a neurodegenerative disease that primarily results from the death of dopaminergic neurons in the substantia nigra [1]. PD progression occurs in six steps, each characterised by the distinct inclusion of Lewy bodies in the somata of the nerve cell [2]. The disease can be categorised into pre-symptomatic and symptomatic phase, currently PD can only be clinically assessed in the late stages [3].  It is believed that approximately 1 in 500 people are affected by PD, it has been found that most people develop symptoms when they are over 50, even though 1 in 20 people who are affected by the condition notice symptoms when they are under 40. Some of the common symptoms of PD include muscular rigidity, shivering of upper and lower limbs or the jaw, speech issues and involuntary movements [4].

Objectives
•	Evaluate the effectiveness of using vocal features in identifying Parkinson’s disease.
•	Use a minimum of two classifications/prediction models on PD.
•	Exploratory data analysis to cross validate approach and assess predictive accuracy.
•	Evaluate the strengths and weaknesses of models.
•	Highlight the possible improvements for future investigations.
Literature review
One of the most discussed methods in which big data has impacted health is via internet of things (IoT). A rise in medical informatics has begun to transform the healthcare system as we know it, reducing costs, removing inefficient processes and most important of all saving lives [5].  Technology-based objective measures (TOMs) have risen in prominence as support in the assessment of motor functions in PD [6]. This has brought about a promise of substantial changes in the monitoring, diagnostics, and therapeutics landscape of PD [7]. 
However, there are still many challenges faced when using big data often due to the variety, volume, and velocity of the biomedical data [4]. Additionally, there are limitations in the sensors (IoT) such as accelerometers and gyroscopes which are designed to detect tremors, bradykinesia, gait impairment and motor complications. On the other hand, data collected from sensors regarding slowness of movement is difficult to infer, this measurement has the potential to be used as a proxy of bradykinesia [8]. Furthermore, there are technological defects which hinder the progression of big data in tackling PD. A majority of the wearable systems are not designed to work in tandem with each other thus, making it awkward to merge data collected by TOMs that are made by different manufacturers [8].  

Data Retrieval and Cleansing
Data Source and Description 
The data set that is being used was found on from the research conducted by Tanas et al [9]. The data set being used has a multivariate set of characteristics, 26 types of attribute values all defined by integers and 5875 instances. This data was obtained from biomedical voice measurements from 42 people with early-stage Parkinson’s disease recruited to a six-month trial where a telemonitoring device would remotely monitor symptom progression. Columns in the table represents: subject number, subject age, subject age, subject gender (0 – male and 1- female), time interval from baseline recruitment date, motor UPDRS, total UPDRS and the 16 biomedical voice measures. The rows represent each one of the 5,875 recordings from the participants. Unified Parkinson’s disease rating scale (UPDRS) is used to track Parkinson’s symptoms and requires patients to be in the clinic and time-consuming examination by staff. UPDRS is made up of three sections: 1 – mentation, behaviour, and mood, 2 – activities of daily living and 3 – motor. The scale for total UPDRS ranges from 0 – 176, with 0 denoting symptom free and 176 describing complete disability. The scale for motor UPDRS ranges from 0 – 108, 0 signifies a healthy individual and 108 denotes severe motor impairment.
Data processing
In the data used there are no missing values however the column “status” has been removed including its labels (0 and 1).  
Feature selection
Once the data set was processed the remaining features are all relevant to the investigation and can be used in machine learning models, statistical analysis, and data visualisation.
Normalising data
Data is normalised for a plethora of reason including removal of duplicates, identifying if the data is normally distributed, removal of outliers and to scale the data whilst maintaining differences between data points. Typically, p-values are used to statistically identify if there are significant differences in the data set. If the p-value is greater than or equal to 0.05 then a significant difference is present. For this investigation, a Min-Max scaler was used to normalise the data.
External libraries
NumPy: Numerical python is a python library that consists of multidimensional array objects and provides access to a large library of math functions that operate on arrays and matrices.
Scikit-learn: A python library that provides supervised and unsupervised learning algorithms. It is built upon other python libraries such as NumPy, pandas and matplotlib.
Matplotlib: This is a python visualization library that is an extension of NumPy. It provides a wide selection of graphs where the axis and scales can be changed in accordance with requirements.
Pandas: This is a python library that is built upon NumPy and is used for data manipulation and analysis.
Min-Max scaler: This normalising method is based on sklean. This method works by scaling the data to a fixed range (typically 0-1).
Methodology
The purpose of this study is to show whether someone has PD. To do this eXtreme gradient boosting which is based on decision trees and regression analysis will be used. Both methods are classification models and will be employed to decipher if someone has PD or not. 
The models that have been used both have benefits and draw backs however based on the data and the aims of the study these are the most suitable methods to use when concluding the effectiveness of non-invasive methods in detecting PD.
eXtreme boosting gradient
XGBoost is a machine learning algorithm used on classification and regression problems. The algorithm operates by generating a prediction model which takes the form of a weak prediction models such as decision trees. XGBoost utilises a stage-wise fashion similar to other boosting techniques and simplifies them which enables optimisation of an arbitrary differential loss function. The unique feature of XGBoost is that it utilises a more regularised model formalization to control over-fitting, which gives it better performance [10]. 

eXtreme boosting gradient justification
A primary object of this investigation is to accurately detect the presence of PD. XGBoost is built upon other prediction methods such as random forests. Ensemble methods such as XGBoost are so powerful because they combine the predictions of multiple machine learning methods to generate predictions with a higher rate of precision. 
Another reason XGBoost was chosen is due to the steps it takes when making predictions, method such as random forests use bootstrap aggregation which averages the results over many decisions in comparison to eXtreme gradient boosting that uses slower steps because it independently makes predictors thus, providing a stronger model.
Random Forests
Random Forest classifier is another example of an ensemble method that is able to train multiple decision trees whilst, bootstrapping followed by aggregation occur [11].  The term bootstrapping indicates that numerous singular decision trees are trained simultaneously on various subsets of training data. One of the functions of bootstrapping is that it ensures that each singular decision tree used in the random forest is unique, thus reducing the overall variance of the random forest classifier.

Random Forest justification
Random Forests algorithm is an effective classification algorithm that is able to classify large data sets with accuracy. The data used in this investigation contains over 5000 data points. By using Random Forests, the data is tree predictors are generators are made where each tree is dependent on the random vector values. This allows a group of “weak learners” to possible combine and produce a “strong learner” [12].

Exploratory data analysis
After subjecting the data set to machine learning algorithms XGBoost and Random Forests, it is time to delve more into the interrelationships between the data set. To do this a variety of data visualisation tools were used such as scatter graphs, bar charts and histogram plots. These plots were made using the libraries matplotlib, pandas and seaborn which are described above. In accordance with the information provided with the data set: Jitter (%), Jitter (Abs), Jitter: RAP, Jitter:PPQ5 and Jitter: DDP are all measures of variation in fundamental frequency. Whereas Shimmer, Shimmer(dB), Shimmer: APQ3, Shimmer: APQ5, Shimmer: APQ11 and Shimmer: DD are measurements of variation in amplitude. Both these measurements are used because they have been found to have clinical value. Voice amplitude is established as the difference between maximum and minimum values within a pitch period. When there are successive cycles that are dissimilar, they are described as Jitter and shimmer. 


From figure 3 it can be interpreted that the highest density of jitter values congregates around the zero mark. Without running any statistical test, it can be seen there is a large difference between the density around the 0 value in comparison to when the jitter value moves away from 0. The kernel density estimation (line on the graph) is used to estimate the probability density function of a random variable. This means that the relative likelihood that the value of a random variable would be equal to that of the sample. It can be seen that the distribution is weighted to the left and does now represent the standard normal distribution.

Once a density plot of the distribution of jitters was plotted, the next step was to examine the interrelationships between jitter measurements. From the bar charts it can be seen that it is a common theme for the most amount of measurements to agregate around the 0 mark and decrease as reading value increases. The scatter graphs demonstrate that there is a positive correlation between the jitter values suggesting that they are linked.

Once visualisation of Jitter values was completed, Shimmer reading values that were collected in the study were then formulated into graphs. From figure 5 it can be seen that shimmers value congregate at a small increase above zero and have a gradual decrease in density from its peak. The kernel density estimation shows that the distribution of the data is weighted to the left and does not resemble a normally distributed dataset.

In the next step a seaborn pair plot was used to identify any interrelations between shimmer values and see if a relationship between points could be identified. From the bar chart it can be seen that the density of values around 0 are the densest and decrease from this peak. A positive correlation can be seen between shimmer values, this is exhibited in the scatter graph. 
Results and Analysis
In this section of the report, I will be delving into the implementation of the machine learning models and their results. Initial results showed that both models have a high precision rate and can successfully decipher if someone has PD or not.
eXtreme boosting gradient
Once the correct libraries were imported and the features have been selected the implementation of this model is relatively straight forward.
From the tests done using this model an accuracy score of 94.8% was calculated. The gradient boosting algorithm behaved as predicted and produced a high rate of accuracy.

Random Forests
Random forests were employed to monitor the difference in prediction accuracy in comparison the gradient boosting algorithm.


From the results obtained from the Random Forests algorithm we can see that it has a lower accuracy rate (92%) than that of XGBoost algorithm. It is also clear that there is a high percentage of misses at 81%.

The heat map above is displaying the values generated by the confusion matrix. From this we can see that there are 5 misses. The random classifier allows us to have a view of the accuracy of the algorithms by comparing the hits and misses.

Discussion
The results collected from the random classifier, Random Forest algorithm and XGBoost algorithm. As predicted the XGBoost algorithm has the highest level of accuracy with a 4% increase in comparison to Random Forest. We can also see that although random forest has a high precision of hits at 96% it is also missing a large number of predictions at 79%. Depending on what metric you are measuring the performance of the random forest algorithm on the value for hit and misses varies however they all average out of 92%.
The strengths of these algorithms is that they produce predictions with high rates of accuracy and thus would allow us to meet the objective of designing an accurate predicator of PD based on vocal recordings.
The weaknesses of using random forests are that they often over fit the data and are unable to predict beyond the training data set.

Conclusion
To conclude I would like to highlight that I stated the objective of this study is:
•	Evaluate the effectiveness of using vocal features in identifying Parkinson’s disease.
•	Use a minimum of two classifications/prediction models on PD.
•	Exploratory data analysis to cross validate approach and assess predictive accuracy.
•	Evaluate the strengths and weaknesses of models.
•	Highlight the possible improvements for future investigations.
Currently all objectives except the final one has been addressed. I will address this objective in my conclusion.
The limitations of the whole study are that I did not statistically analyse the data and therefore cannot say conclusively if there are significant differences in values. Additionally, I do not display a diagram of the Random forest and as a result was not able to identify root node and deduce the key splits in the decision-making process. Also, to make the investigation more applicable to the real world and the clinicians who diagnose patients with PD, I should have statistically linked the features of the data set to the criteria of the unified Parkinson’s disease ratings scale. This would have allowed me to perform regression analysis to identify correlations between the PD scale and vocal features.
To improve the current study for future investigations and a possible direction for future investigations would be to use blockchain as a storage medium, this would provide added security and a platform that is accessible to all parties that need access to the information. 

References
[1]        W. Dauer and S. Przedborski, “Parkinson’s disease: Mechanisms and models,” Neuron. 2003, doi: 10.1016/S0896-6273(03)00568-3.
[2]        H. Braak, E. Ghebremedhin, U. Rüb, H. Bratzke, and K. Del Tredici, “Stages in the development of Parkinson’s disease-related pathology,” Cell and Tissue Research. 2004, doi: 10.1007/s00441-004-0956-9.
[3]        U. Rüb et al., “Parkinson’s disease: The thalamic components of the limbic loop are severely impaired by α-synuclein immunopositive inclusion body pathology,” Neurobiology of Aging, 2002, doi: 10.1016/S0197-4580(01)00269-X.
[4]        M. SenthilarumugamVeilukandammal, S. Nilakanta, B. Ganapathysubramanian, V. Anantharam, A. Kanthasamy, and A. A Willette, “Big Data and Parkinson’s Disease: Exploration, Analyses, and Data Challenges.,” 2018, doi: 10.24251/hicss.2018.352.
[5]        D. V. Dimitrov, “Medical internet of things and big data in healthcare,” Healthcare Informatics Research. 2016, doi: 10.4258/hir.2016.22.3.156.
[6]        G. Di Lazzaro et al., “Technology-Based Objective Measures Detect Subclinical Axial Signs in Untreated, de novo Parkinson’s Disease,” Journal of Parkinson’s Disease, 2020, doi: 10.3233/JPD-191758.
[7]        C. F. Pasluosta, H. Gassner, J. Winkler, J. Klucken, and B. M. Eskofier, “An emerging era in the management of Parkinson’s disease: Wearable technologies and the internet of things,” IEEE Journal of Biomedical and Health Informatics, 2015, doi: 10.1109/JBHI.2015.2461555.

