package ai.greatfree.distributed.cnn.utils;


import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.logging.Logger;

import org.greatfree.exceptions.RemoteReadException;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.convolutional.Conv2d;

import ai.djl.training.GradientCollector;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.util.Pair;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;                 // FIX 1: needed for resize
import ai.greatfree.distributed.cnn.CNNConfig;
import ai.greatfree.distributed.cnn.child.CNNChild;
import ai.greatfree.distributed.cnn.child.app.ChildProfile; // FIX 4: missing import
import ai.greatfree.distributed.cnn.message.CNNParam;
import ai.greatfree.utils.Accumulator;
import ai.greatfree.utils.CNNModelScratch;
import ai.greatfree.utils.StopWatch;
import ai.greatfree.utils.Training;
import ai.greatfree.utils.Functions.voidTwoFunction;

/**
 * CNN counterpart to RNNTools. Holds every utility the CNN trainer and
 * loader need: image loading, batching, layer primitives, param init,
 * gradient clipping, and the training loop.
 */
public final class CNNTools
{
    private final static Logger log = Logger.getLogger("ai.greatfree.distributed.cnn.utils");

    // ============================================================
    // 1) LOADING
    // ============================================================

    public static Pair<List<Pair<NDArray, NDArray>>, List<String>>
    loadTrainingData(ImageFolderLoader loader) throws IOException, Exception
{
if (loader == null) { log.info("2) loader is NULL!!!"); return null; }

// Declare as List, even though we build an ArrayList under the hood.
List<Pair<NDArray, NDArray>> iter = new ArrayList<>();
for (Pair<NDArray, NDArray> b : loader) iter.add(b);
return new Pair<>(iter, loader.getClassNames());
}

    public static Pair<List<Pair<NDArray, Integer>>, List<String>> loadImageFolder(
            String rootDir, int H, int W, int C, boolean normalize, NDManager manager)
            throws IOException
    {
        File root = new File(rootDir);
        File[] classDirs = root.listFiles(File::isDirectory);
        if (classDirs == null || classDirs.length == 0)
            throw new IllegalArgumentException("No class folders in " + rootDir);

        Arrays.sort(classDirs);
        List<String> classNames = new ArrayList<>();
        for (File d : classDirs) classNames.add(d.getName());

        List<Pair<NDArray, Integer>> samples = new ArrayList<>();
        for (int ci = 0; ci < classDirs.length; ci++)
        {
            File[] files = classDirs[ci].listFiles(
                    f -> f.getName().toLowerCase().matches(".*\\.(png|jpg|jpeg|bmp)$"));
            if (files == null) continue;
            for (File f : files)
            {
                NDArray img = preprocess(f, H, W, C, normalize, manager);
                samples.add(new Pair<>(img, ci));
            }
        }
        log.info("Loaded " + samples.size() + " images across "
                 + classNames.size() + " classes.");
        return new Pair<>(samples, classNames);
    }

 // 1. File overload — used by loadImageFolder
    public static NDArray preprocess(File file, int H, int W, int C,
                                     boolean normalize, NDManager manager) throws IOException
    {
        Image img = ImageFactory.getInstance().fromFile(file.toPath());
        NDArray arr = img.toNDArray(manager, Image.Flag.COLOR);
        return shapeAndScale(arr, H, W, C, normalize);
    }

    // 2. BufferedImage overload — used by CNNTrainer.predict(BufferedImage)
    public static NDArray preprocess(BufferedImage image, int H, int W, int C,
                                     boolean normalize, NDManager manager)
    {
        int h = image.getHeight();
        int w = image.getWidth();
        float[] pix = new float[h * w * 3];
        int idx = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int rgb = image.getRGB(x, y);
                pix[idx++] = (rgb >> 16) & 0xFF;
                pix[idx++] = (rgb >>  8) & 0xFF;
                pix[idx++] =  rgb        & 0xFF;
            }
        NDArray arr = manager.create(pix, new Shape(h, w, 3));
        return shapeAndScale(arr, H, W, C, normalize);
    }

    // 3. Shared pipeline
    private static NDArray shapeAndScale(NDArray arr, int H, int W, int C, boolean normalize)
    {
        arr = NDImageUtils.resize(arr, W, H);
        arr = arr.transpose(2, 0, 1);
        if (C == 1) arr = arr.get(new NDIndex("0:1"));
        arr = arr.toType(DataType.FLOAT32, false);
        if (normalize) arr = arr.div(255f);
        return arr;
    }

    public static ArrayList<Pair<NDArray, NDArray>> imageDataIterSequential(
            List<Pair<NDArray, Integer>> samples, int batchSize, NDManager manager)
    {
        ArrayList<Pair<NDArray, NDArray>> batches = new ArrayList<>();
        int n = samples.size();
        int fullBatches = n / batchSize;
        if (fullBatches == 0) return batches;

        long[] shape = samples.get(0).getKey().getShape().getShape();
        int C = (int) shape[0];
        int H = (int) shape[1];
        int W = (int) shape[2];
        int perImage = C * H * W;

        for (int b = 0; b < fullBatches; b++)
        {
            NDArray images = manager.create(
                    new float[batchSize * perImage],
                    new Shape(batchSize, C, H, W));
            long[] labels = new long[batchSize];

            for (int i = 0; i < batchSize; i++)
            {
                Pair<NDArray, Integer> s = samples.get(b * batchSize + i);
                images.set(new NDIndex(i), s.getKey());
                labels[i] = s.getValue();
            }
            batches.add(new Pair<>(images, manager.create(labels)));
        }
        return batches;
    }

    public static ArrayList<Pair<NDArray, NDArray>> imageDataIterRandom(
            List<Pair<NDArray, Integer>> samples, int batchSize, NDManager manager)
    {
        List<Pair<NDArray, Integer>> copy = new ArrayList<>(samples);
        Collections.shuffle(copy, new Random());
        return imageDataIterSequential(copy, batchSize, manager);
    }

    // ============================================================
    // 3) PARAM INIT
    // ============================================================

    private static NDArray normal(Shape shape, Device device, NDManager manager)
    {
        return manager.randomNormal(0f, 0.01f, shape, DataType.FLOAT32, device);
    }

    public static NDList getParams(int inChannels, int[] numFilters,
                                   int[] kernelSizes, int H, int W,
                                   int numClasses, Device device, NDManager manager)
    {
        NDList params = new NDList();
        int c = inChannels;
        int h = H, w = W;
        for (int i = 0; i < numFilters.length; i++)
        {
            int k = kernelSizes[i];
            double std = Math.sqrt(2.0 / (c * k * k));
            NDArray Wc = manager.randomNormal(0f, (float) std,
                    new Shape(numFilters[i], c, k, k), DataType.FLOAT32, device);
            NDArray bc = manager.zeros(new Shape(numFilters[i]), DataType.FLOAT32, device);
            params.add(Wc);
            params.add(bc);
            c = numFilters[i];
            h = h / 2;     // matches the 2x2 max-pool in CNNModelScratch.forward
            w = w / 2;
        }
        NDArray Wf = normal(new Shape(c * h * w, numClasses), device, manager);
        NDArray bf = manager.zeros(new Shape(numClasses), DataType.FLOAT32, device);
        params.add(Wf);
        params.add(bf);
		return params;
    }
    // ============================================================
    // 4) LAYER PRIMITIVES
    //
    // WARNING (Bug 6): getNDArrayInternal() is DJL's internal API. Its
    // signatures have changed across DJL versions. If these fail to compile,
    // either:
    //   (a) use ai.djl.nn.convolutional.Conv2d / Pool2d as a Block, or
    //   (b) implement im2col + matmul here with only public NDArray methods.
    // ============================================================

    /** 2D convolution. x: [N][Cin][H][W], W: [Cout][Cin][k][k]. */
        public static NDArray conv2d(NDArray x, NDArray W, int stride, int padding)
        {
            NDList result = Conv2d.conv2d(
                    x,                              // input  [N][Cin][H][W]
                    W,                              // weight [Cout][Cin][k][k]
                    null,                           // bias — pass null; add it yourself
                    new Shape(stride, stride),      // stride
                    new Shape(padding, padding),    // padding
                    new Shape(1, 1),                // dilation
                    1);                             // groups
            return result.singletonOrThrow();       // Conv2d.conv2d returns an NDList
        }   

        /**
         * Max pooling from primitives. x: [N][C][H][W].
         * Uses only public NDArray operations: slicing, max over axes, set.
         * No DJL version dependencies.
         */
        public static NDArray maxPool2d(NDArray x, int kernel, int stride)
        {
            NDManager manager = x.getManager();
            long N = x.getShape().get(0);
            long C = x.getShape().get(1);
            long H = x.getShape().get(2);
            long W = x.getShape().get(3);

            long outH = (H - kernel) / stride + 1;
            long outW = (W - kernel) / stride + 1;

            NDArray out = manager.zeros(new Shape(N, C, outH, outW));

            for (long i = 0; i < outH; i++)
            {
                for (long j = 0; j < outW; j++)
                {
                    long h0 = i * stride;
                    long w0 = j * stride;
                    NDArray window = x.get(new NDIndex(
                            ":, :, {}:{}, {}:{}",
                            h0, h0 + kernel,
                            w0, w0 + kernel));
                    // max over the last two axes -> [N][C]
                    NDArray pooled = window.max(new int[]{2, 3});
                    out.set(new NDIndex(":, :, {}, {}", i, j), pooled);
                }
            }
            return out;
        }

    /** Fully connected. x: [N][D], W: [D][Cout], b: [Cout]. */
    public static NDArray linear(NDArray x, NDArray W, NDArray b)
    {
        return x.dot(W).add(b);
    }

    // ============================================================
    // 5) PREDICTION
    // ============================================================

    /**
     * FIX 3: probs is [1][numClasses]. Use getFloat(0, cls), not get(cls),
     * which would index the batch axis.
     */
    public static Pair<String, Float> predict(NDArray image, CNNModelScratch net,
                                              List<String> classNames,
                                              Device device, NDManager manager)
    {
        NDArray batch = image.expandDims(0).toDevice(device, false);  // [1][C][H][W]
        NDArray logits = net.forward(batch);
        NDArray probs = logits.softmax(1);
        int cls = (int) probs.argMax(1).getLong(0L);
        float conf = probs.getFloat(0, cls);
        return new Pair<>(classNames.get(cls), conf);
    }

    // ============================================================
    // 6) GRADIENT CLIPPING
    // ============================================================

    private static void associateWithMan(CNNModelScratch net, NDManager manager)
    {
        for (NDArray p : net.params)
        {
            NDArray gradient = p.getGradient();
            if (gradient != null) gradient.attach(manager);
        }
    }

    public static void gradClippingLocally(CNNModelScratch net, int theta)
    {
        double result = 0;
        for (NDArray param : net.params)
        {
            NDArray g = param.getGradient();
            if (g == null) continue;
            result += g.pow(CNNConfig.SQUARE).sum().getFloat();
        }
        double norm = Math.sqrt(result);
        if (norm > theta)
        {
            for (NDArray param : net.params)
            {
                NDArray g = param.getGradient();
                if (g == null) continue;
                g.muli(theta / norm);
            }
        }
    }

    public static double getNorm(NDManager manager, List<CNNParam> params)
    {
        double result = 0;
        for (CNNParam param : params)
        {
            result += NDArray.decode(manager, param.getParams())
                              .pow(CNNConfig.SQUARE).sum().getFloat();
        }
        return Math.sqrt(result);
    }

    public static List<CNNParam> getGradients(CNNModelScratch net)
    {
        List<CNNParam> params = new ArrayList<>();
        for (NDArray param : net.params)
        {
            if (param.getGradient() != null)
                params.add(new CNNParam(param.encode()));
        }
        return params;
    }

    public static void gradClippingGlobally(CNNModelScratch net, int theta)
            throws ClassNotFoundException, RemoteReadException, IOException
    {
        List<CNNParam> params = new ArrayList<>();
        for (NDArray param : net.params)
            params.add(new CNNParam(param.encode()));
        double norm = CNNChild.CLUSTER().getGradientsAggregation(params);
        if (norm > theta)
        {
            for (NDArray param : net.params)
            {
                NDArray g = param.getGradient();
                if (g == null) continue;
                g.muli(theta / norm);
            }
        }
    }

    // ============================================================
    // 7) TRAINING
    // ============================================================

    private static Pair<Double, Double> trainByEpoch(
            CNNModelScratch net, List<Pair<NDArray, NDArray>> trainIter,
            Loss loss, voidTwoFunction<Integer, NDManager> updater,
            Device device, boolean useRandomIter, NDManager manager)
            throws ClassNotFoundException, RemoteReadException, IOException
    {
        StopWatch watch = new StopWatch();
        watch.start();
        Accumulator metric = new Accumulator(CNNConfig.ACCUMULATOR_NUM);
        try (NDManager childManager = manager.newSubManager())
        {
            for (Pair<NDArray, NDArray> batch : trainIter)
            {
                NDArray X = batch.getKey().toDevice(device, true);
                X.attach(childManager);
                NDArray Y = batch.getValue().toDevice(device, true);
                Y.attach(childManager);

                try (GradientCollector gc = manager.getEngine().newGradientCollector())
                {
                    NDArray logits = net.forward(X);
                    NDArray l = loss.evaluate(new NDList(Y), new NDList(logits)).mean();
                    gc.backward(l);

                    int[] preds = logits.argMax(1).toIntArray();
                    int[] ys = Y.toType(DataType.INT32, false).toIntArray();
                    int correct = 0;
                    for (int i = 0; i < ys.length; i++) if (preds[i] == ys[i]) correct++;
                    metric.add(new float[] { l.getFloat(), correct, ys.length });
                }

                associateWithMan(net, childManager);

                if (!ChildProfile.CHILD().isLocalAggregationFinished(CNNConfig.LOCAL_AGGREGATION_TIME))
                    gradClippingLocally(net, CNNConfig.THETA);
                else
                {
                    ChildProfile.CHILD().clearLocalAggregationTimes();
                    gradClippingGlobally(net, CNNConfig.THETA);
                }

                updater.apply(CNNConfig.UPDATE_BATCH_SIZE, childManager);
            }
        }
        double avgLoss = metric.get(0);
        double accuracy = metric.get(2) > 0 ? metric.get(1) / metric.get(2) : 0.0;
        return new Pair<>(avgLoss, accuracy);
    }
    
    /**
     * Walk <dir>/<class>/<file>, read every image file as raw bytes,
     * and return:
     *   key   = (classNames, images)
     *   value = labels
     *
     * classNames is the alphabetical list of class folder names.
     * labels.get(i) is an index into classNames for images.get(i).
     * Both lists are aligned by index.
     */
    public static Pair<Pair<List<String>, List<byte[]>>, List<Integer>>
            readImageFolderAsBytes(File dir) throws IOException
    {
        File[] classDirs = dir.listFiles(File::isDirectory);
        if (classDirs == null || classDirs.length == 0)
            throw new IOException("No class folders in " + dir.getAbsolutePath());

        Arrays.sort(classDirs);   // same ordering as loadImageFolder

        List<String> classNames = new ArrayList<>();
        for (File d : classDirs) classNames.add(d.getName());

        List<byte[]> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        for (int ci = 0; ci < classDirs.length; ci++)
        {
            File[] files = classDirs[ci].listFiles(
                    f -> f.getName().toLowerCase().matches(".*\\.(png|jpg|jpeg|bmp)$"));
            if (files == null) continue;
            for (File f : files)
            {
                images.add(java.nio.file.Files.readAllBytes(f.toPath()));
                labels.add(ci);
            }
        }
        return new Pair<>(new Pair<>(classNames, images), labels);
    }

    public static Function<NDArray, Pair<String, Float>> train(
            CNNModelScratch net, List<Pair<NDArray, NDArray>> trainIter,
            List<String> classNames, int lr, int numEpochs,
            Device device, boolean useRandomIter, NDManager manager)
            throws ClassNotFoundException, RemoteReadException, IOException
    {
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        voidTwoFunction<Integer, NDManager> updater =
                (batchSize, subManager) -> Training.sgd(net.params, lr, batchSize, subManager);

        Function<NDArray, Pair<String, Float>> predictor =
                (image) -> predict(image, net, classNames, device, manager);

        double avgLoss = 0.0, acc = 0.0;
        Pair<Double, Double> pair;
        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            pair = trainByEpoch(net, trainIter, loss, updater, device, useRandomIter, manager);
            avgLoss = pair.getKey();
            acc = pair.getValue();
            System.out.println("----------------------------------------------------------------------------");
            System.out.printf("Epoch = %d, loss = %.4f, accuracy = %.2f%%%n",
                    epoch + 1, avgLoss, 100.0 * acc);
            System.out.println("----------------------------------------------------------------------------");
        }
        System.out.printf("final: loss=%.4f, acc=%.2f%%, device=%s%n",
                avgLoss, 100 * acc, device);
        return predictor;
    }
}