package ai.greatfree.distributed.rnn.message;

import java.util.Set;

import edu.greatfree.cluster.message.ClusterRequest;
import edu.greatfree.cluster.message.RequestType;

/**
 * Created: 09/01/2024, Bing Li
 */
public class CollectGradientsRequest extends ClusterRequest
{
	private static final long serialVersionUID = -7181099095832521649L;
	
	public CollectGradientsRequest(Set<String> childrenKeys)
	{
		super(RequestType.BROADCAST_REQUEST, RNNAppID.COLLECT_GRADIENTS_REQUEST, childrenKeys);
	}
}

