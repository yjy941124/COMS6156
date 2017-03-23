import time
start_time = time.time()
images ='/research/data/imagenet';
vgg = vgg19_trainable.Vgg19();
vgg.build(images);
print("--- %s seconds ---" % (time.time() - start_time))