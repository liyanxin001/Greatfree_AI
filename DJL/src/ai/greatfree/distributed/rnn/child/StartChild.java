package ai.greatfree.distributed.rnn.child;

import java.io.IOException;

import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.util.TerminateSignal;

import ai.greatfree.distributed.rnn.RNNConfig;
import edu.greatfree.cluster.ClusterProfile;
import edu.greatfree.exceptions.TaskAlreadyExistedException;

/**
 * Created: 08/28/2024, Bing Li
 */
final class StartChild
{
	public static void main(String[] args) throws ClassNotFoundException, IOException, RemoteReadException, TaskAlreadyExistedException, InterruptedException
	{
		System.out.println("Child starting up ...");
		RNNChild.CLUSTER().start(ClusterProfile.getLightChildSpec(RNNConfig.TRAINING_ROOT, RNNConfig.TRAINING_CHILD_PORT, RNNConfig.REGISTRY_IP, RNNConfig.REGISTRY_PORT), RNNConfig.TRAINING_COORDINATOR, new RNNChildTask());
		System.out.println("Child started ...");
		TerminateSignal.SIGNAL().waitTermination();
	}
}
