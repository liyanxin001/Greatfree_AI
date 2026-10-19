package ai.greatfree.distributed.rnn.message;

import java.util.List;

import edu.greatfree.multicast.message.MulticastResponse;

/**
 * Created: 09/01/2024, Bing Li
 */
public class CollectGradientsResponse extends MulticastResponse
{
	private static final long serialVersionUID = 6746558468470547224L;
	
	private List<RNNParam> params;

	public CollectGradientsResponse(List<RNNParam> params, String collaboratorKey)
	{
		super(RNNAppID.COLLECT_GRADIENTS_RESPONSE, collaboratorKey);
		this.params = params;
	}

	public List<RNNParam> getParams()
	{
		return this.params;
	}
}
