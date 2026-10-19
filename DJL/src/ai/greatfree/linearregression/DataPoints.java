package ai.greatfree.linearregression;

import ai.djl.ndarray.NDArray;

/**
 * Created: 08/13/2024, Bing Li
 */
public final class DataPoints
{
	private NDArray x, y;
	
	public DataPoints(NDArray x, NDArray y)
	{
		this.x = x;
		this.y = y;
	}

	public NDArray getX()
	{
		return this.x;
	}
	
	public NDArray getY()
	{
		return this.y;
	}
	
	public String toString()
	{
		return this.x.toString() + "\n" + this.y.toString();
	}
}
