library(stats)
library(outliers)
library(readxl)
library(stringr)
library(caTools)
library(car) 
library(carData) 
library(dplyr) 
library("nortest")
library(polycor) 
library(ggplot2)

HV_data <- read.csv("C:\\Users\\ammar\\OneDrive - University of Stirling\\Documents\\University of Stirling\\MSc Big Data\\Dissertation\\Condor Results\\HourVariance results\\HVVresults.csv")
deathrate <- HV_data$deathrate
HV <- factor(HV_data$HourVariance)

levels(HV)

#compute summary statistics by hourVariance - count mean, sd
group_by(HV_data, HV_data$HourVariance) %>%
  summarise(
    count = n(),
    mean = mean(deathrate, na.rm = TRUE),
    sd = sd(deathrate, na.rm = TRUE),
    median = median(deathrate, na.rm = TRUE),
    IQR = IQR(deathrate, na.rm = TRUE)
  )


# Install
if(!require(devtools)) install.packages("devtools")
devtools::install_github("kassambara/ggpubr")
library("ggpubr")

# Box plots
# Plot death rate by hourVariance and color by group
ggboxplot(HV_data, x = "HourVariance", y = "deathrate", 
          ylab = "Death rate", xlab = "Hour Variance")


# Mean plots
# Plot deathrate by hourVariance
# Add error bars: mean_se
# (other values include: mean_sd, mean_ci, median_iqr, ....)
ggline(HV_data, x = "HourVariance", y = "deathrate", 
       add = c("mean_se", "jitter"), 
       ylab = "deathrate", xlab = "HourVariance")

library("gplots")
plotmeans(deathrate ~ HourVariance, data = HV_data, frame = FALSE,
          xlab = "HourVariance", ylab = "deathrate",
          main="Mean Plot with 95% CI") 

# Compute the analysis of variance
res.aov <- aov(deathrate ~ HourVariance, data = HV_data)
# Summary of the analysis
summary(res.aov)

TukeyHSD(res.aov)

#Check the homogeneity of variance assumption
# 1. Homogeneity of variances
plot(res.aov, 1)
#Checking normality assumption
plot(res.aov, 2)
plot(res.aov, 3)
plot(res.aov, 4)
plot(res.aov, 5)
plot(res.aov, 6)
ggqqplot(deathrate)
shapiro.test(HV_data$HourVariance)
hist(HV_data$deathrate, 
     main="Death rate", 
     xlab="Death rate", 
     border="light blue", 
     col="blue", 
     las=1, 
     breaks=5)

library(car)
leveneTest(deathrate ~ HV, data = HV_data)
#From the output above we can see that the p-value is not less than the significance level of 0.05. 
#This means that there is no evidence to suggest that the variance across groups is statistically significantly different. 
#Therefore, we can assume the homogeneity of variances in the different treatment groups.


kruskal.test(deathrate ~ HourVariance, data = HV_data)


