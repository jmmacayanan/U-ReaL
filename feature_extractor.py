import re
import math
import csv
from urllib.parse import urlparse
import dns.resolver
import dns.exception

class URLFeatureExtractor:
    dns_cache = {}
    WHITELIST = set()

    SUSPICIOUS_KEYWORDS = [
        'login', 'secure', 'update', 'free', 'verify', 'account', 'gift', 'bank',
        'confirm', 'password', 'signin', 'click', 'bonus', 'reward', 'offer', 'urgent',
        'win', 'prize', 'limited', 'billing', 'invoice', 'checkout', 'money', 'cash'
    ]

    @staticmethod
    def normalize_domain(domain_or_url):
        """Convert URL or domain to canonical domain (no www, lowercase)"""
        parsed = urlparse(domain_or_url)
        netloc = parsed.netloc if parsed.netloc else parsed.path
        netloc = netloc.lower().split(':')[0]  # remove port
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return netloc

    @staticmethod
    def get_main_domain(domain):
        """Return main domain + TLD, e.g., docs.google.com → google.com"""
        parts = domain.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        return domain

    @staticmethod
    def load_whitelist(csv_path):
        whitelist = set()
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                url = row[1].strip()
                if url:
                    domain = URLFeatureExtractor.normalize_domain(url)
                    whitelist.add(domain)
        URLFeatureExtractor.WHITELIST = whitelist

    def __init__(self, url):
        self.url = url
        self.domain = self.normalize_domain(url)
        self.main_domain = self.get_main_domain(self.domain)

    def is_whitelisted(self):
        # Exact match or known safe subdomain
        if self.domain in URLFeatureExtractor.WHITELIST:
            return True
        if self.main_domain in URLFeatureExtractor.WHITELIST:
            # Only allow trusted subdomains
            allowed_subdomains = ['www', '']
            subdomain = self.domain.replace('.' + self.main_domain, '')
            return subdomain in allowed_subdomains
        return False


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
        if self.is_whitelisted():
            return 0
        weight = 2
        count = sum(self.url.lower().count(word) for word in self.SUSPICIOUS_KEYWORDS)
        return count * weight

    def subdomain_count(self):
        parts = self.domain.split('.')
        return max(len(parts) - 2, 0)

    def tld_length(self):
        parts = self.domain.split('.')
        return len(parts[-1]) if len(parts) > 1 else 0

    def get_dns_info(self):
        if self.is_whitelisted():
            # Skip DNS queries for whitelisted domains
            return (1, 0, 0, 1)

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
                'suspicious_total': self.count_suspicious_words(),
                'subdomain_count': self.subdomain_count(),
                'tld_length': self.tld_length(),
                'url_entropy': self.url_entropy(),
                'has_a': has_a,
                'has_mx': has_mx,
                'has_ns': has_ns,
                'ip_count': ip_count,
                'is_whitelisted': int(self.is_whitelisted())
            }
        except Exception as e:
            print(f"[Feature Extraction Error] URL: {self.url} → {e}")
            return None
