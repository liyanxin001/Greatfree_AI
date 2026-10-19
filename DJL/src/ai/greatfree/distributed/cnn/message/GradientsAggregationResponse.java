package ai.greatfree.distributed.cnn.message;

import org.greatfree.message.ServerMessage;


/**
 * Created: 08/29/2024, Bing Li
 */
public class GradientsAggregationResponse extends ServerMessage
{
	private static final long serialVersionUID = -8711903017431052124L;
	
//	private List<RNNParams> params;
	private double norm;
	private boolean isSucceeded;

//	public GradientsAggregationResponse(List<RNNParams> params)
	public GradientsAggregationResponse(double norm, boolean isSucceeded)
	{
		super(CNNAppID.GRADIENTS_AGGREGATION_RESPONSE);
//		this.params = params;
		this.norm = norm;
		this.isSucceeded = isSucceeded;
	}

	/*
	public List<RNNParams> getParams()
	{
		return this.params;
	}
	*/
	
	public double getNorm()
	{
		return this.norm;
	}
	
	public boolean isSucceeded()
	{
		return this.isSucceeded;
	}
}