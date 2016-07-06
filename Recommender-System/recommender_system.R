#Recommendation Systems
#codeR


#installing gdata package for matrix manipulatipon
install.packages("gdata")
library(gdata)


#-------------------------------------- Importing data -----------------------------------------

Y = read.csv("D:/Data Analytics using R/Project-Recomnender/Y.csv",header = FALSE)    #User-Movie-Ratings file
R = read.csv("D:/Data Analytics using R/Project-Recomnender/R.csv",header = FALSE)    #Logical Ratings Matrix
R = R*1                                                                               #Logical to Binary Conversion 

# Accepting user ratings from a list of movies
movie_list <- read.csv("D:/Data Analytics using R/Project-Recomnender/movie_list.txt", header=FALSE, sep=";")
movie_list = as.matrix(movie_list)

#Converting all data frames into Matrix

Y = as.matrix(Y)                  #Converting Y data frame into a Matrix
R = as.matrix(R)                  #Converting R data frame into a Matrix

#---------------------Data Exploration and Visualization---------------------#

#There are 1682 Movies and 943 Users.
dim(Y)    

#We can see the similarity between first four users using pearson's correlation
cor(Y[,1:4])

#There are ratings from 1 to 5 in our dataset with 0 indicating no rating
ratings = as.vector(Y)
unique_ratings = unique(ratings)
unique_ratings

#frequency of number of ratings
table(ratings[ratings!=0])


#Lets Visualize these ratings using a bar plot
#According to the dataset, a rating equal to 0 indicates a missing value so we can remove
#them from our count

barplot(table(ratings[ratings!=0]),horiz = T,
        main = "Distribution of the Ratings",
        xlab="Count",ylab="Ratings", col = "orange",xlim = c(0,40000))


#Exploring most viewed movies
most_viewed_movies = rowSums(Y!=0)
most_viewed_movies = most_viewed_movies[order(most_viewed_movies, decreasing =T)]
pie(most_viewed_movies[1:6],labels = movie_list[most_viewed_movies[1:6]])


#Exploring the average ratings
Y_NA = replace(Y,Y==0,NA)
mean_per_movie = rowMeans(Y_NA,na.rm=T)
plot(density(mean_per_movie), main="Average movie ratings")
polygon(density(mean_per_movie),col="blue")



#------------------------------Visualization End-------------------------------#




#-----------------------------------Modeling-----------------------------------#
# A function to calculate the squared error using least squares regression method

calcCost = function(params,Y, R, num_users, num_movies,num_features,lambda=0)
{
  X = matrix(params[1:num_movies*num_features],num_movies, num_features)     #Unrolling values from params into X
  Weights = matrix(params[16821:26260],num_users, num_features)             #Unrolling values from params into Theta
  cost = 0                                                                #Intitializing J = 0, where J is the cost
  squared_error = (X%*%t(Weights) - Y)*R                                   #Calulate squared error 
  # Applying regularization to reduce overfitting of the data
  reg_Weights = lambda/2 * sum(sum(Weights^2))        
  reg_X = lambda/2 * sum(sum(X^2))
  # Calculating final cost to be minimized
  cost = 1/2 * sum(sum(squared_error)) + reg_Weights + reg_X
  return(cost)
}

# Below function calculates the gradient to minimize the above calcCost function and obtain parameter values
calcGradient = function(params,Y, R, num_users, num_movies,num_features,lambda=0)
{
  X = matrix(params[1:num_movies*num_features],num_movies, num_features)            #Unrolling values from params into X
  Weights = matrix(params[16821:26260],num_users, num_features)                 #Unrolling values from params into Theta
  X_grad = matrix(nrow = nrow(X),ncol = ncol(X))                       # an empty matrix of size X
  Weights_grad = matrix(nrow = nrow(Weights),ncol = ncol(Weights))     # an empty matrix of size Weights
  errors = (X%*%t(Weights) - Y)*R                                      #error term
  X_grad = errors%*%Weights +lambda * X                                #including regularization parameter for X_grad
  Weights_grad = t(errors)%*%X + lambda * Weights                     #including regularization parameter for Weights_grad
  t = unmatrix(X_grad,byrow = F)                    #unrolling X_grad into t
  t1 = unmatrix(Weights_grad,byrow = F)             #unrolling Weights_grad into t1
  gradient = c(t,t1)
  return(gradient)                                  #return gradient
}


##My individual ratings to group of movies where rating in index 1 corresponds to movie 1 rating
my_ratings = matrix(nrow = 1682, ncol = 1)

my_ratings[1]=4
my_ratings[98] = 2
my_ratings[7] = 3
my_ratings[12]= 5
my_ratings[54] = 4
my_ratings[64]= 5
my_ratings[66]= 3
my_ratings[69] = 5
my_ratings[183] = 4
my_ratings[226] = 5
my_ratings[355]= 5
my_ratings[is.na(my_ratings)]<-0
my_ratings
paste("New user rating:")
paste("Rated ",my_ratings[1],"for ",movie_list[1])
paste("Rated ",my_ratings[98],"for ",movie_list[98])
paste("Rated ",my_ratings[7],"for ",movie_list[7])
paste("Rated ",my_ratings[12],"for ",movie_list[12])
paste("Rated ",my_ratings[54],"for ",movie_list[54])
paste("Rated ",my_ratings[64],"for ",movie_list[64])
paste("Rated ",my_ratings[66],"for ",movie_list[66])
paste("Rated ",my_ratings[69],"for ",movie_list[69])
paste("Rated ",my_ratings[183],"for ",movie_list[183])
paste("Rated ",my_ratings[226],"for ",movie_list[226])
paste("Rated ",my_ratings[355],"for ",movie_list[355])


##Adding new user rating to this Y matrix of movie-user
Y = cbind(Y,my_ratings)
##Creating R new column by 1-0 rule from my_ratings
R = cbind(R,my_ratings)
R[,944] = ifelse(my_ratings!=0,1,0)

##Function to normalize ratings
normalize_ratings = function(Y)
{
  mean_per_movie = as.numeric(mean_per_movie)
  Y_temp1 = replace(Y,Y==0,NA)
  temp = Y_temp1-mean_per_movie                    #subtracting each Y rating with mean per movie
  temp[is.na(temp)]=0
  Ynorm = as.matrix(temp)
  return(Ynorm)
}

Ynorm = normalize_ratings(Y)                 # Calculate Y normalized

##-------------------------Random initialization of X and Theta ---------------------------------


##Set initial parameters
X = matrix(rnorm(num_movies*num_features,mean = 0,sd = 1),num_movies,num_features)
Weights = matrix(rnorm(num_users*num_features,mean = 0,sd = 1),num_users,num_features)
num_movies=dim(Y)[1]
num_users = dim(Y)[2]
num_features = dim(X)[2]
##Unrolling X and Theta into par to use in OPTIM

X_new = unmatrix(X,byrow = F)
Weights_new = unmatrix(Weights,byrow = F)
params = c(X_new,Weights_new)
params = as.vector(params)


##Optimization of params using Optim. This function is used to minimize the cost in 
##calcCost function to obtain best parameters for linear regression

params_final = optim(params,calcCost,calcGradient,method = 'L-BFGS-B',lower = 0,upper = 1,
      params,Ynorm,num_users, num_movies,num_features,lambda)

#Obtaining optimized parameter values
params_final=params_final$par

#unrolling params into X_final and Weights_final
X_final = matrix(params_final[1:num_movies*num_features],num_movies,num_features)
Weights_final = matrix(params_final[16821:26260],num_users,num_features)


##-------------------------------------Algorithm Learning complete -------------------------------

# Predicted Values in P
P = X_final%*%t(Weights_final)

# adding mean values for our new user so that he can atleast be recommended most famous movies
my_prediction = P[,944]+ mean_per_movie

# Filtering ratings above 5
my_prediction = ifelse(my_prediction<5,my_prediction,5)

# calculating top recommended movies
Top_recommend=sort(my_prediction,decreasing = T)
index_of_movie = order(my_prediction,decreasing = T)


# Recommending 10 best movies to the new user
cat("Top 10 movies you should also see are: \n" ,movie_list[index_of_movie[1:10]])
paste(movie_list[index_of_movie[1:10]])
  



#------------------------------------ End of file ------------------------------------------










