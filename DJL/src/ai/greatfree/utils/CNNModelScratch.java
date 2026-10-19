package ai.greatfree.utils;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;

/**
 * Created: 10/06/2026
 *
 * A CNN Model implemented from scratch. Counterpart to RNNModelScratch.
 * The math (conv2d, pool, linear) is injected via function hooks so this
 * class stays pure scaffolding.
 */
public class CNNModelScratch
{
    // ---- Architecture hyperparameters ----
    public int inChannels;
    public int imageHeight;
    public int imageWidth;
    public int numClasses;
    public int[] numFilters;      // e.g. {8, 16}
    public int[] kernelSizes;     // e.g. {3, 3}

    // ---- Learned parameters ----
    // Layout: [conv1_W, conv1_b, conv2_W, conv2_b, ..., fc_W, fc_b]
    public NDList params;

    // ---- Injected function hooks (the actual math lives in CNNTools) ----
    // (x, W, stride, padding) -> feature map
    public Functions.QuadFunction<NDArray, NDArray, Integer, Integer, NDArray> convFn;
    // (x, kernel, stride, unused) -> pooled map
    public Functions.QuadFunction<NDArray, Integer, Integer, Integer, NDArray> poolFn;
    // (x, W, b) -> logits
    public Functions.TriFunction<NDArray, NDArray, NDArray, NDArray> linearFn;

    public CNNModelScratch(
            int numClasses,
            int inChannels,
            int imageHeight,
            int imageWidth,
            int[] numFilters,
            int[] kernelSizes,
            Device device,
            Functions.TriFunction<Integer, Integer, Device, NDList> getParams,
            Functions.QuadFunction<NDArray, NDArray, Integer, Integer, NDArray> convFn,
            Functions.QuadFunction<NDArray, Integer, Integer, Integer, NDArray> poolFn,
            Functions.TriFunction<NDArray, NDArray, NDArray, NDArray> linearFn)
    {
        this.numClasses  = numClasses;
        this.inChannels  = inChannels;
        this.imageHeight = imageHeight;
        this.imageWidth  = imageWidth;
        this.numFilters  = numFilters;
        this.kernelSizes = kernelSizes;

        // Params allocated once, via the injected initializer
        // (in the real code, CNNTools.getParams takes more args; wrap it in a closure)
        this.params = getParams.apply(inChannels, numFilters.length, device);

        this.convFn   = convFn;
        this.poolFn   = poolFn;
        this.linearFn = linearFn;
    }

    /**
     * Forward pass for one batch.
     * Input  X:  [N][C][H][W]
     * Output Y:  [N][numClasses]  (raw logits — softmax applied by the loss)
     */
    public NDArray forward(NDArray X)
    {
        // Walk through the param list: each conv layer consumes (W, b),
        // then relu + pool. At the end, a single linear layer.
        NDArray x = X;
        int p = 0;

        for (int i = 0; i < numFilters.length; i++)
        {
            NDArray W = params.get(p++);
            NDArray b = params.get(p++);

            x = convFn.apply(x, W, 1, 1);           // stride 1, pad 1 (same size)
            x = x.add(b.reshape(1, -1, 1, 1));      // broadcast bias over H,W
            x = Activation.relu(x);                     // activation
            x = poolFn.apply(x, 2, 2, 0);           // 2x2 max pool -> halves H,W
        }

        // Flatten [N][C][H][W] -> [N][C*H*W]
        int N = (int) x.getShape().get(0);
        x = x.reshape(N, -1);

        // Final linear layer: [N][C*H*W] -> [N][numClasses]
        NDArray Wf = params.get(p++);
        NDArray bf = params.get(p);
        return linearFn.apply(x, Wf, bf);
    }
}
