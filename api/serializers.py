from rest_framework import serializers

class resultSerializer(serializers.Serializer):
    result = serializers.CharField(max_length=30)