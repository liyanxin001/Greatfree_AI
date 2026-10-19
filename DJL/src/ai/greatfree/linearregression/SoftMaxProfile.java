package ai.greatfree.linearregression;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

/**
 * The class is created by myself. 08/15/2024, Bing Li
 * 
 * Created: 08/14/2024, Bing Li
 */
final public class SoftMaxProfile
{
	private int numInputs;
	private int numOutputs;
	private float lr;
	private int numEpochs;
	private NDList params;
	private NDManager manager;
	private String figurePrefix;
	
	private SoftMaxProfile()
	{
		this.numInputs = 784;
		this.numOutputs = 10;
		this.lr = 0.1f;
		this.numEpochs = 5;
		this.figurePrefix = "id-";
	}
	
	private static SoftMaxProfile instance = new SoftMaxProfile();
	
	public static SoftMaxProfile TRAIN()
	{
		if (instance == null)
		{
			instance = new SoftMaxProfile();
			return instance;
		}
		else
		{
			return instance;
		}
	}

	public void setParams(NDList params)
	{
		this.params = params;
	}
	
	public NDList getParams()
	{
		return this.params;
	}
	
	public void setManager(NDManager manager)
	{
		this.manager = manager;
	}
	
	public NDManager getManager()
	{
		return this.manager;
	}
	
	public int getNumInputs()
	{
		return this.numInputs;
	}
	
	public int getNumOutputs()
	{
		return this.numOutputs;
	}
	
	public float getLR()
	{
		return this.lr;
	}
	
	public int getNumEpochs()
	{
		return this.numEpochs;
	}
	
	public String getFigurePrefix()
	{
		return this.figurePrefix;
	}
}
