package ai.greatfree.mlp;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.loss.Loss;

/**
 * Created: 08/16/2024, Bing Li
 */
final class NonLinearFuns
{
	public static NDArray relu(NDArray x)
	{
		return x.maximum(0f);
	}

	public static NDArray net(NDArray x, NDArray w1, NDArray w2, NDArray b1, NDArray b2, int numInputs)
	{
		x = x.reshape(new Shape(-1, numInputs));
		NDArray H = relu(x.dot(w1).add(b1));
		return H.dot(w2).add(b2);
	}
	
	public static Loss getLoss()
	{
		return Loss.softmaxCrossEntropyLoss();
	}
}
