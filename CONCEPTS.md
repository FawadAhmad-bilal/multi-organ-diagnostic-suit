# Concept Notes: Grad-CAM & Transfer Learning Consistency

These are the internals behind this project, written up for learning
purposes — not required reading to use the app, but useful if you want to
understand or rebuild the explainability layer yourself.

## Why Grad-CAM needs the *last convolutional layer*, not any layer

A CNN classifier is basically two halves:

```
Input -> [convolutional feature extractor] -> [GlobalAveragePooling] -> [Dense classifier head] -> softmax
```

Early conv layers detect edges and textures — generic, not class-specific.
By the *last* conv layer, the feature maps have become fairly abstract and
class-relevant (e.g. "this map lights up for rounded opacities," "this one
lights up for irregular lesion borders"). Grad-CAM asks: **which of those
last-layer feature maps mattered most for the class the model picked, and
where in the image did they activate?** That's why the technique always
targets the layer right before pooling — it's the last point where spatial
location (which pixel) is still preserved before everything gets flattened
into a single vector.

## What the gradient actually means here

`tape.gradient(class_channel, conv_output)` computes, for every pixel of
every feature map, "if this value increased slightly, how much would the
predicted class's score increase?" A big positive gradient at some spot in
some channel means the model's confidence is sensitive to that spot — i.e.
it's using it as evidence. Averaging that gradient over each channel gives
one importance weight per channel (`pooled_grads`), and multiplying each
channel by its weight before summing is literally computing a weighted
vote across channels: "how much did the important channels light up
*here*."

## Why the ReLU step matters

After weighting and summing, `tf.maximum(heatmap, 0)` throws away negative
values. Negative means "this suppressed the predicted class" — useful for
different analyses, but for a standard Grad-CAM visualization you only want
regions that pushed *toward* the prediction, since that's what a viewer
reads as "why the model chose this class."

## Why preprocessing must match training exactly

Every backbone (ResNet50, MobileNetV2, EfficientNet, VGG16) was originally
pretrained on ImageNet with its own specific pixel normalization
(`preprocess_input` from each `keras.applications.*` submodule). If your
training pipeline used a given backbone's `preprocess_input`, the app at
inference time must use the exact same function, or the input distribution
the classifier head sees will be shifted from what it learned on — this is
a silent bug that doesn't crash anything, it just quietly produces wrong
predictions. If predictions look off for a specific module, this mismatch
is the first thing to check — not the model itself.

## Why the app keeps two copies of the image array

`preprocess_image()` returns both a preprocessed batch (for the model) and
a raw 0-255 array (for the Grad-CAM overlay). If you fed the
ImageNet-normalized array into the overlay function, the colors/contrast
displayed to the user would look wrong even though the model's prediction
is unaffected — normalization is only needed by the network, never by a
human looking at the picture.

## Rebuilding this yourself for a new model

To build a Grad-CAM from scratch for a new model, you only need to know
three things about it: (1) the name of its last conv/activation layer
before pooling, (2) the exact preprocessing function it was trained with,
and (3) its input size. Everything else in `grad_cam.py` is generic and
does not change between architectures — that's the whole point of writing
it once as a utility instead of once per model.
