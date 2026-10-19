package ai.greatfree.linearregression;

import ai.djl.ndarray.NDList;

/**
 * Created: 08/15/2024, Bing Li
 */
final class ActivationFunction
{
	public static NDList softmax(NDList arrays)
	{
		return new NDList(arrays.singletonOrThrow().logSoftmax(1));
	}
}
