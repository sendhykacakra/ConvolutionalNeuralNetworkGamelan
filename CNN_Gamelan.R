# Load packages
library(keras)
library(EBImage)

# Set wd where images are located
setwd("X://CNN_Gamelan/image/")
# Set d where to save images
save_in <- "X://CNN_Gamelan/imagenew/"
# Load images names
images <- list.files()
images
# Set width
w <- 100
# Set height
h <- 100

# Main loop resize images and set them to greyscale
for(i in 1:length(images))
{
  # Try-catch is necessary since some images
  # may not work.
  result <- tryCatch({
    # Image name
    imgname <- images[i]
    # Read image
    img <- readImage(imgname)
    # Resize image 28x28
    img_resized <- resize(img, w = w, h = h)
    # Path to file
    path <- paste(save_in, imgname, sep = "")
    # Save image
    writeImage(img_resized, path, quality = 70)
    # Print status
    print(paste("Done",i,sep = " "))},
    # Error function
    error = function(e){print(e)})
}


setwd("X://CNN_Gamelan/imagenew/")
# Read Images
images <- list.files()
images

summary(images)

list_of_images = lapply(images, readImage ) 
list_of_images
display(list_of_images[[15]])
#create train
train <- list_of_images[c(1:20,26:45,51:70)]
str(train)
display(train[[1]])

#create test
test <- list_of_images[c(21:25,46:50,71:75)]
test
display(test[[1]])

par(mfrow = c(6, 10))
for (i in 1:60) plot(train[[i]])

# Resize & combine
str(train)
for (i in 1:60) {train[[i]] <- resize(train[[i]], 100, 100)}
for (i in 1:15) {test[[i]] <- resize(test[[i]], 100, 100)}

train <- combine(train)
str(train)
x <- tile(train, 5)
display(x, title='Pictures')

test <- combine(test)
y <- tile(test, 3)
display(y, title = 'Pics')

# Reorder dimension
train <- aperm(train, c(4, 1, 2, 3))
test <- aperm(test, c(4, 1, 2, 3))
str(train)

# Response
trainy <- c(rep(0,20),rep(1,20),rep(2,20))
testy <- c(rep(0,5),rep(1,5),rep(2,5))

# One hot encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

# Model
model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = c(100, 100, 3)) %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate=0.25) %>%
  layer_dense(units = 3, activation = 'softmax') %>%
  
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_sgd(lr = 0.01,
                                    decay = 1e-6,
                                    momentum = 0.9,
                                    nesterov = T),
          metrics = c('accuracy'))
summary(model)

# Fit model
history <- model %>%
  fit(train,
      trainLabels,
      epochs = 45,
      batch_size = 32,
      validation_split = 0.2,
      validation_data = list(test, testLabels))
plot(history)

# Evaluation & Prediction - train data
model %>% evaluate(train, trainLabels)
pred <- model %>% predict_classes(train)
table(Predicted = pred, Actual = trainy)

prob <- model %>% predict_proba(train)
cbind(prob, Predicted_class = pred, Actual = trainy)

# Evaluation & Prediction - test data
model %>% evaluate(test, testLabels)
pred <- model %>% predict_classes(test)
table(Predicted = pred, Actual = testy)

prob <- model %>% predict_proba(test)
cbind(prob, Predicted_class = pred, Actual = testy)

#save model
save_model_weights_hdf5(model,filepath='X://CNN_Gamelan/modelcnngamelan.hdf5',overwrite=TRUE)
model=load_model_weights_hdf5(model,filepath="X://CNN_Gamelan/modelcnngamelan.hdf5",by_name=FALSE)

####################
##########very fixxx
img <- c("b1.jpg","g1.jpg","k1.jpg")
image <- list()
for (i in 1:3) {image[[i]] <- readImage(img[i])}

display(image[[2]])
# Get the image as a matrix

for (i in 1:3) {image[[i]] <- resize(image[[i]], 100, 100)}

fixx <- combine(image)
y <- tile(fixx, 3)
display(y, title = 'Pics')

str(fixx)
got <- aperm(fixx, c(4, 1, 2, 3))
str(got)

testy <- c(0, 1, 2)

# One hot encoding
testLabels <- to_categorical(testy)

pred <- model %>% predict_classes(got)
table(Predicted = pred, Actual = testy)

prob <- model %>% predict_proba(got)
cbind(prob, Predicted_class = pred, Actual = testy)


