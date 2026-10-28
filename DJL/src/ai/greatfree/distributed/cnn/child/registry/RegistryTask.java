package ai.greatfree.distributed.cnn.child.registry;

/*     */ import java.util.Calendar;
/*     */ import java.util.logging.Logger;
/*     */ import org.greatfree.exceptions.PeerKeyNotExistedInRegistryException;
/*     */ import org.greatfree.framework.container.p2p.message.ChatPartnerRequest;
/*     */ import org.greatfree.framework.container.p2p.message.ChatRegistryRequest;
/*     */ import org.greatfree.framework.container.p2p.message.ClusterIPRequest;
/*     */ import org.greatfree.framework.container.p2p.message.IsRootOnlineRequest;
/*     */ import org.greatfree.framework.container.p2p.message.LeaveClusterNotification;
/*     */ import org.greatfree.framework.container.p2p.message.MulticastChildrenRequest;
/*     */ import org.greatfree.framework.container.p2p.message.PartnersRequest;
/*     */ import org.greatfree.framework.container.p2p.message.PeerAddressRequest;
/*     */ import org.greatfree.framework.container.p2p.message.PeerDisableStateRequest;
/*     */ import org.greatfree.framework.container.p2p.message.PortRequest;
/*     */ import org.greatfree.framework.container.p2p.message.RegisterPeerRequest;
/*     */ import org.greatfree.framework.container.p2p.message.SearchBrokerRequest;
/*     */ import org.greatfree.framework.container.p2p.message.UnregisterPeerRequest;
/*     */ import org.greatfree.message.ServerMessage;
/*     */ import org.greatfree.message.container.Notification;
/*     */ import org.greatfree.message.container.Request;
/*     */ import org.greatfree.server.container.ServerTask;
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ public final class RegistryTask
/*     */   implements ServerTask
/*     */ {
/*  32 */   private static final Logger log = Logger.getLogger("org.greatfree.framework.container.p2p.registry");
/*     */ 
/*     */ 
/*     */   
/*     */   public void processNotification(Notification notification) {
/*  37 */     switch (notification.getApplicationID()) {
/*     */       
/*     */       case 83:
/*  40 */         log.info("LEAVE_CLUSTER_NOTIFICATION received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  41 */         Register.leaveCluster((LeaveClusterNotification)notification);
/*     */         break;
/*     */       
/*     */       case 99:
/*  45 */         log.info("SHUTDOWN_SERVER_NOTIFICATION received @" + String.valueOf(Calendar.getInstance().getTime()));
/*     */         
/*     */         try {
/*  48 */           RegisterServer.CS().stop(3000L);
/*     */         }
/*  50 */         catch (ClassNotFoundException|java.io.IOException|InterruptedException|org.greatfree.exceptions.RemoteReadException e) {
/*     */           
/*  52 */           e.printStackTrace();
/*     */         } 
/*     */         break;
/*     */     } 
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public ServerMessage processRequest(Request request) {
/*  62 */     switch (request.getApplicationID()) {
/*     */       
/*     */       case 75:
/*  65 */         log.info("PEER_CHAT_REGISTRY_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  66 */         return Register.registerChat((ChatRegistryRequest)request);
/*     */       
/*     */       case 77:
/*  69 */         log.info("PEER_CHAT_PARTNER_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  70 */         return Register.checkChatPartner((ChatPartnerRequest)request);
/*     */       
/*     */       case 84:
/*  73 */         log.info("PARTNERS_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  74 */         return Register.checkPartners((PartnersRequest)request);
/*     */       
/*     */       case 5:
/*  77 */         log.info("REGISTER_PEER_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  78 */         return Register.registerPeer((RegisterPeerRequest)request);
/*     */       
/*     */       case 7:
/*  81 */         log.info("PORT_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  82 */         return Register.retrievePort((PortRequest)request);
/*     */       
/*     */       case 9:
/*  85 */         log.info("CLUSTER_IP_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  86 */         return Register.retrieveClusterIP((ClusterIPRequest)request);
/*     */       
/*     */       case 122:
/*  89 */         log.info("MULTICAST_CHILDREN_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  90 */         return Register.retrieveMulticastChildrenIPs((MulticastChildrenRequest)request);
/*     */       
/*     */       case 13:
/*  93 */         log.info("PEER_ADDRESS_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  94 */         return Register.retrievePeerAddress((PeerAddressRequest)request);
/*     */       
/*     */       case 124:
/*  97 */         log.info("PEER_DISABLE_STATE_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*  98 */         return Register.retrievePeerDisableState((PeerDisableStateRequest)request);
/*     */       
/*     */       case 126:
/* 101 */         log.info("SEARCH_BROKER_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/* 102 */         return Register.retrieveBroker((SearchBrokerRequest)request);
/*     */       
/*     */       case 86:
/* 105 */         log.info("ALL_REGISTERED_IPS_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/* 106 */         return Register.retrieveAllRegisteredIPs();
/*     */       
/*     */       case 88:
/* 109 */         log.info("ALL_PEER_NAMES_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/* 110 */         return Register.retrieveAllPeerNames();
/*     */       
/*     */       case 79:
/* 113 */         log.info("IS_ROOT_ONLINE_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/* 114 */         return Register.isRootOnline((IsRootOnlineRequest)request);
/*     */       
/*     */       case 11:
/* 117 */         log.info("UNREGISTER_PEER_REQUEST received @" + String.valueOf(Calendar.getInstance().getTime()));
/*     */         
/*     */         try {
/* 120 */           return Register.unregisterPeer((UnregisterPeerRequest)request);
/*     */         }
/* 122 */         catch (PeerKeyNotExistedInRegistryException e) {
/*     */           
/* 124 */           log.info(e.toString()); break;
/*     */         } 
/*     */     } 
/* 127 */     return null;
/*     */   }
/*     */ }
