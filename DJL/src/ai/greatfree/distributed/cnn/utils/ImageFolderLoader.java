package ai.greatfree.distributed.cnn.utils;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.greatfree.util.Builder;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;

/**
 * CNN counterpart to SequentialTextLoader.
 * Loads images from <root>/<class>/<file>, batches them into
 * (images [B][C][H][W], labels [B]) pairs.
 */
public class ImageFolderLoader implements Iterable<Pair<NDArray, NDArray>>
{
    // ---- Data produced by the loader ----
    private final ArrayList<Pair<NDArray, NDArray>> dataIter;   // list of batches
    private final List<String> classNames;                      // index -> class name
    private final List<Pair<NDArray, Integer>> samples;         // all (image, label) before batching

    private final int batchSize;
    private final int imageHeight;
    private final int imageWidth;
    private final int channels;
    private final boolean normalize;
    private final boolean shuffle;

    public ImageFolderLoader(ImageFolderLoaderBuilder builder) throws Exception
    {
        this.batchSize    = builder.getBatchSize();
        this.imageHeight  = builder.getImageHeight();
        this.imageWidth   = builder.getImageWidth();
        this.channels     = builder.getChannels();
        this.normalize    = builder.isNormalize();
        this.shuffle      = builder.isShuffle();
        NDManager manager = builder.getManager();

        // 1) Scan directory -> (image tensor, class index) samples + class names
        Pair<List<Pair<NDArray, Integer>>, List<String>> scan =
                CNNTools.loadImageFolder(
                        builder.getRootDir(),
                        this.imageHeight, this.imageWidth, this.channels,
                        this.normalize, manager);

        this.samples     = scan.getKey();
        this.classNames  = scan.getValue();

        // 2) Batch them
        if (this.shuffle)
        {
            this.dataIter = CNNTools.imageDataIterRandom(
                    this.samples, this.batchSize, manager);
        }
        else
        {
            this.dataIter = CNNTools.imageDataIterSequential(
                    this.samples, this.batchSize, manager);
        }
    }

    public List<String> getClassNames()
    {
        return this.classNames;
    }

    public List<Pair<NDArray, Integer>> getSamples()
    {
        return this.samples;
    }

    // ---- Fluent builder, mirrors SequentialTextLoaderBuilder ----
    public static class ImageFolderLoaderBuilder implements Builder<ImageFolderLoader>
    {
        private String rootDir;
        private int batchSize;
        private int imageHeight;
        private int imageWidth;
        private int channels;
        private boolean normalize;
        private boolean shuffle;
        private NDManager manager;

        public ImageFolderLoaderBuilder() {}

        public ImageFolderLoaderBuilder rootDir(String rootDir)
        {
            this.rootDir = rootDir;
            return this;
        }

        public ImageFolderLoaderBuilder batchSize(int batchSize)
        {
            this.batchSize = batchSize;
            return this;
        }

        public ImageFolderLoaderBuilder imageHeight(int imageHeight)
        {
            this.imageHeight = imageHeight;
            return this;
        }

        public ImageFolderLoaderBuilder imageWidth(int imageWidth)
        {
            this.imageWidth = imageWidth;
            return this;
        }

        public ImageFolderLoaderBuilder channels(int channels)
        {
            this.channels = channels;
            return this;
        }

        public ImageFolderLoaderBuilder normalize(boolean normalize)
        {
            this.normalize = normalize;
            return this;
        }

        /** Equivalent of the RNN's useRandomIter: shuffle samples before batching. */
        public ImageFolderLoaderBuilder shuffle(boolean shuffle)
        {
            this.shuffle = shuffle;
            return this;
        }

        public ImageFolderLoaderBuilder manager(NDManager manager)
        {
            this.manager = manager;
            return this;
        }

        @Override
        public ImageFolderLoader build()
        {
            try
            {
                return new ImageFolderLoader(this);
            }
            catch (Exception e)
            {
                e.printStackTrace();
                return null;
            }
        }

        public String getRootDir()      { return this.rootDir; }
        public int getBatchSize()       { return this.batchSize; }
        public int getImageHeight()     { return this.imageHeight; }
        public int getImageWidth()      { return this.imageWidth; }
        public int getChannels()        { return this.channels; }
        public boolean isNormalize()    { return this.normalize; }
        public boolean isShuffle()      { return this.shuffle; }
        public NDManager getManager()   { return this.manager; }
    }

    @Override
    public Iterator<Pair<NDArray, NDArray>> iterator()
    {
        return this.dataIter.iterator();
    }
}
