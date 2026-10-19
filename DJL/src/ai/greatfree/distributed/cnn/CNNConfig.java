package ai.greatfree.distributed.cnn;

import org.greatfree.util.UtilConfig;

/**
 * CNN counterpart to RNNConfig. Static constants only — no logic.
 */
public final class CNNConfig
{
    // ============================================================
    // 1. PATHS  (same root, different subfolders)
    // ============================================================
    public final static String HOME_PATH          = "/home/liyanxin/AI/Data/";
    public final static String SOURCE_DATA_PATH   = HOME_PATH + "SourceData/";
    public final static String TRAINING_DATA_PATH = HOME_PATH + "TrainingData/";
    public final static String TRAINED_DATA_PATH  = HOME_PATH + "TrainedData/";

    // ============================================================
    // 2. IDENTITY / SENTINELS  (identical to RNNConfig)
    // ============================================================
    public final static String CHILD    = "Child";
    public final static String NO_PATH  = UtilConfig.EMPTY_STRING;
    public final static String NO_FILE  = UtilConfig.EMPTY_STRING;

    // ============================================================
    // 3. DATA & MODEL HYPERPARAMETERS
    //    (replaces RNN's NUM_HIDDENS / TRAINING_NUM_STEPS / MAX_TOKENS)
    // ============================================================
    public final static int TRAINING_BATCH_SIZE = 32;
    public final static boolean USE_RANDOM_ITERATIVE = true;

    // Image geometry — every image is resized to this
    public final static int IMAGE_HEIGHT   = 32;
    public final static int IMAGE_WIDTH    = 32;
    public final static int IMAGE_CHANNELS = 3;      // 3 = RGB, 1 = grayscale

    // Normalization — divide pixels by 255 to get [0,1]
    public final static boolean NORMALIZE_0_1 = true;

    // Architecture — list of conv layers, each with a filter count and kernel size
    public final static int[] NUM_FILTERS = { 8, 16, 32 };   // conv1: 8, conv2: 16, conv3: 32
    public final static int[] KERNEL_SIZES = { 3, 3, 3 };    // 3x3 kernels for all
    public final static int   POOL_SIZE    = 2;              // 2x2 max pool
    public final static int   POOL_STRIDE  = 2;
    public final static int   NUM_CLASSES  = 10;             // or infer from folder count

    // Class-name source (folder names become labels)
    public final static String CLASS_FOLDER_NAME = "class";

    // Data split
    public final static int LOAD_START_INDEX = 0;
    public final static int LOAD_END_INDEX   = 10000;

    // ============================================================
    // 4. TRAINING SCHEDULE  (identical shape to RNNConfig)
    // ============================================================
    public final static int NUM_OF_EPOCHS  = 100;
    public final static int LEARNING_RATE  = 1;

    public final static String MAX_EPOCH   = "MAX_EPOCH";
    public final static int    NUM_EPOCHS  = 500;

    // (RNNConfig has NUM_PERIODS for text generation; CNN doesn't need it —
    //  you predict a single class per image, not a sequence.)

    // ============================================================
    // 5. DISTRIBUTED TRAINING & GRADIENT CLIPPING
    //    (nearly identical to RNNConfig — these knobs are architecture-agnostic)
    // ============================================================
    public final static int ACCUMULATOR_NUM       = 3;   // was 2 for loss+count; now loss+correct+total
    public final static int THETA                 = 1;
    public final static int SQUARE                = 2;
    public final static int LOCAL_AGGREGATION_TIME = 5;
    public final static int UPDATE_BATCH_SIZE     = 1;

    public final static double NO_NORM = -1d;

    public final static String TRAINING_COORDINATOR      = "TrainingCoordinator";
    public final static int    TRAINING_COORDINATOR_PORT = 9000;
    public final static String TRAINING_ROOT             = "TrainingRoot";
    public final static int    TRAINING_ROOT_PORT        = 9050;
    public final static int    TRAINING_CHILD_PORT       = 9080;

    public final static String REGISTRY_IP   = "192.168.1.25";
    public final static int    REGISTRY_PORT = 8941;
}
