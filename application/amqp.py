import pika
import logging

logger = logging.getLogger(__name__)

def connect_to_amqp(amqp_ip, amqp_user, amqp_password, amqp_vhost):    

    # Otherwise, establish a new connection for this process
    connection_successful = True
    try:
        credentials = pika.PlainCredentials(amqp_user, amqp_password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=amqp_ip,
                virtual_host=amqp_vhost,
                credentials=credentials,
                connection_attempts=5,
                retry_delay=5,
                socket_timeout=600
            )
        )
        channel = connection.channel()
    
    except Exception as e:
        connection_successful = False
        logger.error(f"failed to connect", e)
    
    return connection_successful, connection, channel

def create_queue(channel, queue_name, dlx=None, dlx_queue='deadletters', is_exclusive=False, is_auto_delete=False):
    
    # Declare the queue with 'dlx' as the DLX if provided
    if dlx:
        result = channel.queue_declare(queue=queue_name, exclusive=is_exclusive, auto_delete=is_auto_delete, arguments={
            'x-dead-letter-exchange': dlx,
            'x-dead-letter-routing-key': dlx_queue
        })
    else:
        result = channel.queue_declare(queue=queue_name, exclusive=is_exclusive, auto_delete=is_auto_delete)

    return result.method.queue

def create_exchange(channel, exchange_name, exchange_type='direct'):
    channel.exchange_declare(exchange=exchange_name, exchange_type=exchange_type)

def bind_queue_to_exchange(channel, queue_name, exchange_name, routing_key=None):
    channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=routing_key)

def become_consumer(channel, queue_name, callback_function):
    channel.basic_consume(queue=queue_name, on_message_callback=callback_function, auto_ack=False)
    channel.start_consuming()

def send_message_to_exchange(channel, exchange_name, routing_key, message, headers=None):
    properties = pika.BasicProperties(delivery_mode=2)  # make message persistent
    if headers is not None:
        properties.headers = headers

    channel.basic_publish(exchange=exchange_name,
                          routing_key=routing_key,
                          body=message,
                          properties=properties)
