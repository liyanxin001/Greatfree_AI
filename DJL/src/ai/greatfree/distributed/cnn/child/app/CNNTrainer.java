package ai.greatfree.distributed.cnn.child.app;


import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.function.Function;
import java.util.logging.Logger;

import org.greatfree.exceptions.RemoteReadException;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;
import ai.greatfree.distributed.cnn.CNNConfig;
import ai.greatfree.distributed.cnn.child.CNNChild;
import ai.greatfree.distributed.cnn.utils.CNNTools;
import ai.greatfree.distributed.cnn.utils.ImageFolderLoader;
import ai.greatfree.utils.CNNModelScratch;
import ai.greatfree.utils.Functions.QuadFunction;
import ai.greatfree.utils.Functions.TriFunction;

public final class CNNTrainer
{
    private NDManager manager;
    private ImageFolderLoader loader;
    private Pair<List<Pair<NDArray, NDArray>>, List<String>> dataset;
    private List<Pair<NDArray, NDArray>> trainIter;
    private List<String> classNames;

    // ---- Function hooks — types now match CNNTools' actual signatures ----
    // FIX 1, 2, 3: NDList -> NDArray; poolFn's second arg is not NDList.
    private TriFunction<Integer, Integer, Device, NDList> getParamsFn;
    private QuadFunction<NDArray, NDArray, Integer, Integer, NDArray> convFn;
    private QuadFunction<NDArray, Integer, Integer, Integer, NDArray> poolFn;
    private TriFunction<NDArray, NDArray, NDArray, NDArray>          fcFn;

    private Device device;
    private int numEpochs;
    private CNNModelScratch net;
    private Function<NDArray, Pair<String, Float>> predictor;

    private CNNTrainer() {}

    private static CNNTrainer instance = new CNNTrainer();

    public static CNNTrainer CHILD()
    {
        return instance;   // the null check was redundant; instance is initialized above
    }

    public void dispose()
    {
        this.manager.close();
    }

    private final static Logger log = Logger.getLogger("ai.greatfree.distributed.cnn.child.app");

    public void init() throws IOException, Exception
    {
        this.manager = NDManager.newBaseManager();

        // ---- 1) Data: images laid out as  <root>/<class>/<file>.png ----
        this.loader = new ImageFolderLoader.ImageFolderLoaderBuilder()
                .rootDir(ChildProfile.CHILD().getTrainingDataPath())
                .imageHeight(CNNConfig.IMAGE_HEIGHT)
                .imageWidth(CNNConfig.IMAGE_WIDTH)
                .channels(CNNConfig.IMAGE_CHANNELS)
                .batchSize(CNNConfig.TRAINING_BATCH_SIZE)
                .shuffle(CNNConfig.USE_RANDOM_ITERATIVE)
                .normalize(CNNConfig.NORMALIZE_0_1)
                .manager(this.manager)
                .build();

        if (this.loader == null)
        {
            log.info("1) loader is NULL!!!");
            return;
        }

        this.dataset     = CNNTools.loadTrainingData(this.loader);
        this.trainIter   = this.dataset.getKey();
        this.classNames  = this.dataset.getValue();

        // ---- 2) Function hooks ----
        // FIX 4: capture the missing args (filters, kernels, dims, numClasses, manager)
        //        so the TriFunction<Integer,Integer,Device,NDList> contract is satisfied.
        final int numClasses = this.classNames.size();
        this.getParamsFn = (inCh, numLayers, dev) ->
                CNNTools.getParams(
                        inCh,
                        CNNConfig.NUM_FILTERS,
                        CNNConfig.KERNEL_SIZES,
                        CNNConfig.IMAGE_HEIGHT,
                        CNNConfig.IMAGE_WIDTH,
                        numClasses,
                        dev,
                        this.manager);

        this.convFn = (x, W, stride, pad) -> CNNTools.conv2d(x, W, stride, pad);
        this.poolFn = (x, k, stride, ignored) -> CNNTools.maxPool2d(x, k, stride);
        this.fcFn   = (x, W, b) -> CNNTools.linear(x, W, b);

        this.device = this.manager.getDevice();

        this.numEpochs = Integer.getInteger(CNNConfig.MAX_EPOCH, CNNConfig.NUM_EPOCHS);

        // ---- 3) Build the model ----
        this.net = new CNNModelScratch(
                numClasses,
                CNNConfig.IMAGE_CHANNELS,
                CNNConfig.IMAGE_HEIGHT,
                CNNConfig.IMAGE_WIDTH,
                CNNConfig.NUM_FILTERS,
                CNNConfig.KERNEL_SIZES,
                this.device,
                this.getParamsFn,
                this.convFn,
                this.poolFn,
                this.fcFn
        );
    }

    public CNNModelScratch getCNN()
    {
        return this.net;
    }

    public void train() throws ClassNotFoundException, RemoteReadException, IOException, InterruptedException
    {
        this.predictor = CNNTools.train(
                this.net,
                this.trainIter,
                this.classNames,
                CNNConfig.LEARNING_RATE,
                this.numEpochs,
                this.device,
                CNNConfig.USE_RANDOM_ITERATIVE,
                this.manager);

        CNNChild.CLUSTER().notifyTrainingDone();
    }


    public Pair<String, Float> predict(BufferedImage image) throws IOException
    {
        NDArray arr = CNNTools.preprocess(
                image,
                CNNConfig.IMAGE_HEIGHT,
                CNNConfig.IMAGE_WIDTH,
                CNNConfig.IMAGE_CHANNELS,
                CNNConfig.NORMALIZE_0_1,
                this.manager);
        return this.predictor.apply(arr);
    }
}
