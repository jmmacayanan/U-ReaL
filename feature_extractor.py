import re
import math
from urllib.parse import urlparse
import dns.resolver
import dns.exception


class URLFeatureExtractor:
    dns_cache = {}

    def __init__(self, url):
        self.url = url
        self.domain = self.extract_domain()

    def has_ip(self):
        return int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', self.url)))

    def count_dots(self):
        return self.url.count('.')

    def count_hyphens(self):
        return self.url.count('-')

    def url_length(self):
        return len(self.url)

    def url_entropy(self):
        prob = [float(self.url.count(c)) / len(self.url) for c in set(self.url)]
        entropy = -sum([p * math.log2(p) for p in prob])
        return entropy

    def count_suspicious_words(self):
        keywords = ['login', 'secure', 'update', 'free', 'verify', 'account', 'gift', 'bank']
        return sum(self.url.lower().count(word) for word in keywords)

    def extract_domain(self):
        try:
            return urlparse(self.url).netloc
        except:
            return ''

    def subdomain_count(self):
        parts = self.domain.split('.')
        return max(len(parts) - 2, 0)

    def tld_length(self):
        parts = self.domain.split('.')
        return len(parts[-1]) if len(parts) > 1 else 0

    def get_dns_info(self):
        if self.domain in URLFeatureExtractor.dns_cache:
            return URLFeatureExtractor.dns_cache[self.domain]

        has_a = has_mx = has_ns = False
        ip_count = 0

        try:
            answers = dns.resolver.resolve(self.domain, 'A', lifetime=1)
            ip_count = len(answers)
            has_a = True
        except dns.exception.DNSException:
            pass

        try:
            dns.resolver.resolve(self.domain, 'MX', lifetime=1)
            has_mx = True
        except dns.exception.DNSException:
            pass

        try:
            dns.resolver.resolve(self.domain, 'NS', lifetime=1)
            has_ns = True
        except dns.exception.DNSException:
            pass

        result = (int(has_a), int(has_mx), int(has_ns), ip_count)
        URLFeatureExtractor.dns_cache[self.domain] = result
        return result

    def extract_features(self):
        try:
            has_a, has_mx, has_ns, ip_count = self.get_dns_info()

            return {
                'url_len': self.url_length(),
                'dot_count': self.count_dots(),
                'hyphen_count': self.count_hyphens(),
                'has_ip': self.has_ip(),
                'suspicious_words': self.count_suspicious_words(),
                'subdomain_count': self.subdomain_count(),
                'tld_length': self.tld_length(),
                'url_entropy': self.url_entropy(),
                'has_a': has_a,
                'has_mx': has_mx,
                'has_ns': has_ns,
                'ip_count': ip_count
            }
        except Exception as e:
            print(f"[Feature Extraction Error] URL: {self.url} → {e}")
            return None
