package ai.greatfree.linearregression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/14/2024, Bing Li
 */
class Net
{
//	public static NDArray net(NDArray x, NDList params)
	public static NDArray net(NDArray x)
	{
		NDArray currentW = SoftMaxProfile.TRAIN().getParams().get(0);
		NDArray currentB = SoftMaxProfile.TRAIN().getParams().get(1);
		return LNTraining.softmax(x.reshape(new Shape(-1, SoftMaxProfile.TRAIN().getNumInputs())).dot(currentW).add(currentB));
	}
}
