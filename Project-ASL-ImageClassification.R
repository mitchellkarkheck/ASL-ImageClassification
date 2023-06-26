setwd("C:/")

install.packages("reticulate")
install.packages("keras")
install.packages("tensorflow")
install_tensorflow(extra_packages="pillow")
install_keras(extra_packages="scipy")
install.packages("reticulate")


library(reticulate)
library(keras)
library(tensorflow)


#keras dataset of black and white photos of numbers 0-9
data<-dataset_mnist()

#separating train and test file
train_x<-data$train$x
train_y<-data$train$y
test_x<-data$test$x
test_y<-data$test$y

rm(data)

# converting a 2D array into a 1D array for feeding into the MLP and normalising the matrix
train_x <- array(train_x, dim = c(dim(train_x)[1], prod(dim(train_x)[-1]))) / 255
test_x <- array(test_x, dim = c(dim(test_x)[1], prod(dim(test_x)[-1]))) / 255


#converting the target variable to one hot encoded vectors using keras inbuilt function
train_y<-to_categorical(train_y,10)
test_y<-to_categorical(test_y,10)
#defining a keras sequential model
model <- keras_model_sequential()

#defining the model with 1 fully connected linear input layer, 784 in and out channels, 1 hidden layer[784 neurons] with dropout rate 0.4 and 1 output layer[10 neurons]
#i.e number of digits from 0 to 9

model %>%
  layer_dense(units = 784, input_shape = 784) %>%
  layer_dropout(rate=0.4)%>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 10) %>%
  layer_activation(activation = 'softmax')

#compiling the defined model with metric as accuracy and optimizer as adam.
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

#fitting the model on the training dataset
model1 = model %>% fit(train_x, train_y, epochs = 50, batch_size = 128)

#Evaluating model on the cross validation dataset
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)


setwd("C:/Users/mkark/OneDrive - HEC Montreal/Documents/HEC/adv stat learning/Image Classification Project/archive/imagedata")

#In R, set your working directory to the folder where all the images are located. With the dir() command, we get a list of all birds in the correct order and save it for later purposes:
label_list <- dir("C:/Users/mkark/Desktop/imagedata")
output_n <- length(label_list)
save(label_list, file="label_list.R")

#set the target size to which all images will be rescaled (in pixels):
width <- 224
height<- 224
target_size <- c(width, height)
rgb <- 3 #color channels

#After we set the path to the training data, we use the image_data_generator() function to define the preprocessing of the data. We could, for instance, pass further arguments for data augmentation
#(apply small amounts of random blurs or rotations to the images to add variation to the data and prevent overfitting)
#to this function, but lets just rescale the pixel values to values between 0 and 1 and tell the function to reserve 20% of the data for a validation dataset:

path_train = "C:/Users/mkark/Desktop/imagedatatrain"
train_data_gen <- image_data_generator(rescale = 1/255, validation_split = 0.2)

train_images <- flow_images_from_directory(path_train,
                                           train_data_gen,
                                           subset = 'training',
                                           target_size = target_size,
                                           class_mode = "categorical",
                                           shuffle=F,
                                           classes = label_list,
                                           seed = 2021)


validation_images <- flow_images_from_directory(path_train,
                                                train_data_gen, 
                                                subset = 'validation',
                                                target_size = target_size,
                                                class_mode = "categorical",
                                                classes = label_list,
                                                seed = 2021)
table(train_images$classes)

#loading exception from imagenet, pre-trained on millions of images and 20 k categories
mod_base <- application_xception(weights = 'imagenet', 
                                 include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base) #freezing weight matrices from exception

model_function <- function(learning_rate = 0.001, 
                           dropoutrate=0.2, n_dense=1024){
  
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    mod_base %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = n_dense) %>%
    layer_activation("relu") %>%
    layer_dropout(dropoutrate) %>%
    layer_dense(units=output_n, activation="softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = learning_rate),
    metrics = "accuracy"
  )
  
  return(model)
  
}


model <- model_function()
model


batch_size <- 32
epochs <- 6
hist <- model %>% fit_generator(
  train_images,
  steps_per_epoch = train_images$n %/% batch_size, 
  epochs = epochs, 
  validation_data = validation_images,
  validation_steps = validation_images$n %/% batch_size,
  verbose = 2
)



path_test <- "C:/Users/mkark/Desktop/imagedatavalid"
test_data_gen <- image_data_generator(rescale = 1/255)
test_images <- flow_images_from_directory(path_test,
                                          test_data_gen,
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = label_list,
                                          shuffle = F,
                                          seed = 2021)

table(test_images$classes)

model %>% evaluate_generator(test_images, 
                             steps = test_images$n/batch_size)


# visualize the digits
par(mfcol=c(1,5))
pic1 = plot(as.raster(train_images[[1]][[1]][18,,,]))
pic2 = plot(as.raster(train_images[[12]][[1]][19,,,]))
pic3 = plot(as.raster(train_images[[20]][[1]][30,,,]))
pic4 = plot(as.raster(train_images[[40]][[1]][22,,,]))
pic5 = plot(as.raster(train_images[[27]][[1]][17,,,]))






