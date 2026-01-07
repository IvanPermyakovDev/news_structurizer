import pika
from typing import Optional


class RabbitMQProducer:
    def __init__(
        self,
        host: str = "rabbitmq",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        queue_name: str = "recordings",
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.queue_name = queue_name
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.BlockingChannel] = None
        self._connect()

    def _connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self._connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
        )
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=self.queue_name, durable=True)

    def publish(self, message: str):
        if self._connection.is_closed:
            self._connect()
        self._channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=message,
            properties=pika.BasicProperties(delivery_mode=2),
        )

    def close(self):
        if self._connection and self._connection.is_open:
            self._connection.close()
