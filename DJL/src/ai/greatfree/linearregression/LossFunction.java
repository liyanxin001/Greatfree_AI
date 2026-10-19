package ai.greatfree.linearregression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;

/**
 * Created: 08/14/2024, Bing Li
 */
class LossFunction
{
	public static NDArray crossEntropy(NDArray yHat, NDArray y)
	{
		// Here, y is not guranteed to be of datatype int or long
		// and in our case we know its a float32.
		// We must first convert it to int or long(here we choose int)
		// before we can use it with NDIndex to "pick" indices.
		// It also takes in a boolean for returning a copy of the existing NDArray
		// but we don't want that so we pass in `false`.
		NDIndex pickIndex = new NDIndex().addAllDim(Math.floorMod(-1, yHat.getShape().dimension())).addPickDim(y);
		return yHat.get(pickIndex).log().neg();
	}
}
