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

MD_data <- read.csv("C:\\Users\\ammar\\OneDrive - University of Stirling\\Documents\\University of Stirling\\MSc Big Data\\Dissertation\\Condor Results\\missDose results\\MDresults.csv")
deathrate <- MD_data$deathrate
MD <- factor(MD_data$missDose)

levels(MD)

#compute summary statistics by hourVariance - count mean, sd
group_by(MD_data, HV_data$missDose) %>%
  summarise(
    count = n(),
    mean = mean(deathrate, na.rm = TRUE),
    sd = sd(deathrate, na.rm = TRUE),
    median = median(deathrate, na.rm = TRUE),
    IQR = IQR(deathrate, na.rm = TRUE)
  )


library("ggpubr")

# Box plots
# Plot death rate by hourVariance and color by group
ggboxplot(MD_data, x = "missDose", y = "deathrate", 
          ylab = "Death rate", xlab = "Miss Dose")


# Mean plots
# Plot deathrate by hourVariance
# Add error bars: mean_se
# (other values include: mean_sd, mean_ci, median_iqr, ....)
ggline(HV_data, x = "missDose", y = "deathrate", 
       add = c("mean_se", "jitter"), 
       ylab = "deathrate", xlab = "Miss Dose")

library("gplots")
plotmeans(deathrate ~ missDose, data = MD_data, frame = FALSE,
          xlab = "missDose", ylab = "deathrate",
          main="Mean Plot with 95% CI") 

# Compute the analysis of variance
res.aov <- aov(deathrate ~ missDose, data = MD_data)
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

ggqqplot(deathrate)
shapiro.test(MD_data$missDose)
hist(HV_data$deathrate, 
     main="Death rate", 
     xlab="Death rate", 
     border="light blue", 
     col="blue", 
     las=1, 
     breaks=5)

library(car)
leveneTest(deathrate ~ MD, data =MD_data)
#From the output above we can see that the p-value is not less than the significance level of 0.05. 
#This means that there is no evidence to suggest that the variance across groups is statistically significantly different. 
#Therefore, we can assume the homogeneity of variances in the different treatment groups.


kruskal.test(deathrate ~ missDose, data = MD_data)


